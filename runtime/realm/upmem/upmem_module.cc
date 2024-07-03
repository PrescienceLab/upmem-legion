/* Copyright 2024 Stanford University, NVIDIA Corporation
 *                Los Alamos National Laboratory, Northwestern University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "realm/upmem/upmem_module.h"

#include "realm/cmdline.h"
#include "realm/event_impl.h"
#include "realm/idx_impl.h"
#include "realm/logging.h"
#include "realm/tasks.h"

#include "realm/mutex.h"
#include "realm/utils.h"

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

namespace Realm {

namespace Upmem {
    Logger log_dpu("upmem");
    Logger log_stream("upmemstream");
    Logger log_dpudma("upmemdma");
    Logger log_upmemipc("upmemipc");

    ////////////////////////////////////////////////////////////////////////
    //
    // class UpmemModuleConfig

    UpmemModuleConfig::UpmemModuleConfig(void)
        : ModuleConfig("upmem")
    {
    }

    void UpmemModuleConfig::configure_from_cmdline(std::vector<std::string>& cmdline)
    {
        // read command line parameters
        CommandLineParser cp;

        cp.add_option_int("-ll:tasklets", cfg_tasklets);
        cp.add_option_int("-ll:num_dpus", cfg_num_dpus);

        bool ok = cp.parse_command_line(cmdline);
        if (!ok) {
            log_dpu.fatal() << "error reading Upmem command line parameters";
            assert(false);
        }
        if (cfg_tasklets != 0) {
            printf("Running with %d Tasklets per DPU\n", cfg_tasklets);
        }
    }
    ////////////////////////////////////////////////////////////////////////
    //
    // class UpmemModule

    UpmemModule* upmem_module_singleton = 0;

    UpmemModule::UpmemModule(RuntimeImpl* _runtime)
        : Module("upmem")
        , config(nullptr)
        , runtime(_runtime)
        , shared_worker(0)
        , upmemipc_condvar(upmemipc_mutex)
        , upmemipc_responses_needed(0)
        , upmemipc_releases_needed(0)
        , upmemipc_exports_remaining(0)
    {
        assert(!upmem_module_singleton);
        upmem_module_singleton = this;
        rh_listener = new DPUReplHeapListener(this);
    }

    UpmemModule::~UpmemModule(void)
    {
        assert(config != nullptr);
        config = nullptr;
        assert(upmem_module_singleton == this);
        upmem_module_singleton = 0;
        delete rh_listener;
    }

    /*static*/ ModuleConfig* UpmemModule::create_module_config(RuntimeImpl* runtime)
    {
        UpmemModuleConfig* config = new UpmemModuleConfig();
        return config;
    }

    /*static*/ Module* UpmemModule::create_module(RuntimeImpl* runtime)
    {
        // create a module to fill in with stuff
        UpmemModule* m = new UpmemModule(runtime);

        UpmemModuleConfig* config = checked_cast<UpmemModuleConfig*>(runtime->get_module_config("upmem"));
        assert(config != nullptr);
        assert(config->finish_configured);
        assert(m->name == config->get_name());
        assert(m->config == nullptr);
        m->config = config;

        return m;
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void UpmemModule::initialize(RuntimeImpl* runtime)
    {
        Module::initialize(runtime);
        std::vector<unsigned> fixed_indices;

        // each DPU has 64MB of MRAM and 64KB of cache
        // we need to add each DPU to the core reservation in Realm

        dpus.resize(config->cfg_num_dpus);
        dpu_info.resize(config->cfg_num_dpus);

        unsigned dpu_count = 0;
        // try to get cfg_num_dpus, working through the list in order
        for (size_t i = config->cfg_skip_dpu_count;
             (i < dpu_info.size()) && (static_cast<int>(dpu_count) < config->cfg_num_dpus);
             i++) {
            int idx = (fixed_indices.empty() ? i : fixed_indices[i]);

            DPUWorker* worker;
            worker = new DPUWorker;
            DPU* g = new DPU(this, dpu_info[idx], worker, idx);
            dedicated_workers[g] = worker;
            dpus[dpu_count++] = g;
        }

        if (static_cast<int>(dpu_count) < config->cfg_num_dpus) {
            log_dpu.fatal() << config->cfg_num_dpus << " DPUs requested, but only "
                            << dpu_count << " available!";
            assert(false);
        }
        runtime->repl_heap.add_listener(rh_listener);
    }

    // create any memories provided by this module (default == do nothing)
    //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
    void UpmemModule::create_memories(RuntimeImpl* runtime)
    {
        Module::create_memories(runtime);
        // each DPU has its own memory
        if (config->cfg_mram_mem_size > 0) {
            for (std::vector<DPU*>::iterator it = dpus.begin(); it != dpus.end(); it++) {
                (*it)->create_mram_memory(runtime, config->cfg_mram_mem_size);
            }
        }
    }

    // create any processors provided by the module (default == do nothing)
    //  (each new ProcessorImpl should use a Processor from
    //   RuntimeImpl::next_local_processor_id)
    void UpmemModule::create_processors(RuntimeImpl* runtime)
    {
        Module::create_processors(runtime);
        // each DPU has 64MB of MRAM and 64KB of cache
        // we can load a MAX of 64MB per DPU. This is the stack size limit here.
        for (std::vector<DPU*>::iterator it = dpus.begin(); it != dpus.end(); it++) {
            // each dpu in the dpu iterator is a processor in Realm.
            (*it)->create_processor(runtime, 64 * MEGABYTE);
        }
    }

    // create any DMA channels provided by the module (default == do nothing)
    void UpmemModule::create_dma_channels(RuntimeImpl* runtime)
    {
        Module::create_dma_channels(runtime);

        // if we don't have any mram memory, we can't do any DMAs
        // if(!mram)
        //   return;

        // r->add_dma_channel(new DPUChannel(this, XFER_DPU_IN_MRAM, &r->bgwork));
        // r->add_dma_channel(new DPUfillChannel(this, &r->bgwork));
        // // r->add_dma_channel(new DPUreduceChannel(this, &r->bgwork));

        // if(!pinned_sysmems.empty()) {
        //   r->add_dma_channel(new DPUChannel(this, XFER_DPU_TO_MRAM, &r->bgwork));
        //   r->add_dma_channel(new DPUChannel(this, XFER_DPU_FROM_MRAM, &r->bgwork));
        // } else {
        //   log_dpu.warning() << "DPU " << proc->me << " has no accessible system memories!?";
        // }
        // // only create a p2p channel if we have peers (and an fb)
        // if(!peer_fbs.empty() || !upmemipc_mappings.empty()) {
        //   r->add_dma_channel(new DPUChannel(this, XFER_DPU_PEER_MRAM, &r->bgwork));
        // }
    }

    // create any code translators provided by the module (default == do nothing)
    void UpmemModule::create_code_translators(RuntimeImpl* runtime)
    {
        Module::create_code_translators(runtime);
    }

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void UpmemModule::cleanup(void) { Module::cleanup(); }

    struct dpu_set_t* UpmemModule::get_task_upmem_stream()
    {
        // if we're not in a dpu task, this'll be null
        if (ThreadLocal::current_dpu_stream)
            return ThreadLocal::current_dpu_stream->get_stream();
        else
            return 0;
    }

}; // namespace Upmem

}; // namespace Realm
