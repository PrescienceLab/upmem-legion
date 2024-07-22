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
    {}

    void UpmemModuleConfig::configure_from_cmdline(std::vector<std::string> &cmdline)
    {
      // read command line parameters
      CommandLineParser cp;

      cp.add_option_int("-ll:tasklets", cfg_tasklets);
      cp.add_option_int("-ll:num_dpus", cfg_num_dpus);

      bool ok = cp.parse_command_line(cmdline);
      if(!ok) {
        log_dpu.fatal() << "error reading Upmem command line parameters";
        assert(false);
      }
      if(cfg_tasklets != 0) {
        printf("Running with %d Tasklets per DPU\n", cfg_tasklets);
      }
    }
    ////////////////////////////////////////////////////////////////////////
    //
    // class UpmemModule

    UpmemModule *upmem_module_singleton = 0;

    UpmemModule::UpmemModule(RuntimeImpl *_runtime)
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

    /*static*/ ModuleConfig *UpmemModule::create_module_config(RuntimeImpl *runtime)
    {
      UpmemModuleConfig *config = new UpmemModuleConfig();
      return config;
    }

    /*static*/ Module *UpmemModule::create_module(RuntimeImpl *runtime)
    {
      // create a module to fill in with stuff
      UpmemModule *m = new UpmemModule(runtime);

      UpmemModuleConfig *config =
          checked_cast<UpmemModuleConfig *>(runtime->get_module_config("upmem"));
      assert(config != nullptr);
      assert(config->finish_configured);
      assert(m->name == config->get_name());
      assert(m->config == nullptr);
      m->config = config;

      return m;
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void UpmemModule::initialize(RuntimeImpl *runtime)
    {
      Module::initialize(runtime);
      std::vector<unsigned> fixed_indices;

      // each DPU has 64MB of MRAM and 64KB of cache
      // we need to add each DPU to the core reservation in Realm

      dpus.resize(config->cfg_num_dpus);
      dpu_info.resize(config->cfg_num_dpus);

      unsigned dpu_count = 0;
      // try to get cfg_num_dpus, working through the list in order
      for(size_t i = config->cfg_skip_dpu_count;
          (i < dpu_info.size()) && (static_cast<int>(dpu_count) < config->cfg_num_dpus);
          i++) {
        int idx = (fixed_indices.empty() ? i : fixed_indices[i]);

        DPUWorker *worker;
        worker = new DPUWorker;
        DPU *g = new DPU(this, dpu_info[idx], worker, idx);
        dedicated_workers[g] = worker;
        dpus[dpu_count++] = g;
      }

      if(static_cast<int>(dpu_count) < config->cfg_num_dpus) {
        log_dpu.fatal() << config->cfg_num_dpus << " DPUs requested, but only "
                        << dpu_count << " available!";
        assert(false);
      }
      runtime->repl_heap.add_listener(rh_listener);
    }

    // create any memories provided by this module (default == do nothing)
    //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
    void UpmemModule::create_memories(RuntimeImpl *runtime)
    {
      Module::create_memories(runtime);
      // each DPU has its own memory
      if(config->cfg_mram_mem_size > 0) {
        for(std::vector<DPU *>::iterator it = dpus.begin(); it != dpus.end(); it++) {
          (*it)->create_mram_memory(runtime, config->cfg_mram_mem_size);
        }
      }
    }

    // create any processors provided by the module (default == do nothing)
    //  (each new ProcessorImpl should use a Processor from
    //   RuntimeImpl::next_local_processor_id)
    void UpmemModule::create_processors(RuntimeImpl *runtime)
    {
      Module::create_processors(runtime);
      // each DPU has 64MB of MRAM and 64KB of cache
      // we can load a MAX of 64MB per DPU. This is the stack size limit here.
      for(std::vector<DPU *>::iterator it = dpus.begin(); it != dpus.end(); it++) {
        // each dpu in the dpu iterator is a processor in Realm.
        (*it)->create_processor(runtime, 64 * MEGABYTE);
      }
    }

    // create any DMA channels provided by the module (default == do nothing)
    void UpmemModule::create_dma_channels(RuntimeImpl *runtime)
    {
      // before we create dma channels, see how many of the system memory ranges
      //  we can register with Upmem
      if(config->cfg_pin_sysmem && !dpus.empty()) {
        const std::vector<MemoryImpl *> &local_mems =
            runtime->nodes[Network::my_node_id].memories;
        std::vector<MemoryImpl *> all_local_mems;
        all_local_mems.insert(all_local_mems.end(), local_mems.begin(), local_mems.end());
        // </NEW_DMA>
        for(std::vector<MemoryImpl *>::iterator it = all_local_mems.begin();
            it != all_local_mems.end(); it++) {
          // ignore MRAM memories or anything that doesn't have a "direct" pointer
          if(((*it)->kind == MemoryImpl::MKIND_MRAM))
            continue;

          // skip any memory that's over the max size limit for host
          //  registration
          if((config->cfg_hostreg_limit > 0) &&
             ((*it)->size > config->cfg_hostreg_limit)) {
            log_dpu.info() << "memory " << (*it)->me << " is larger than hostreg limit ("
                           << (*it)->size << " > " << config->cfg_hostreg_limit
                           << ") - skipping registration";
            continue;
          }

          void *base = (*it)->get_direct_ptr(0, (*it)->size);
          if(base == 0)
            continue;

          // // using GPU 0's context, attempt a portable registration
          // hipError_t ret;
          // {
          //   ret = hipHostRegister(base, (*it)->size,
          //         hipHostRegisterPortable |
          //         hipHostRegisterMapped);
          // }
          // if(ret != hipSuccess) {
          //   log_dpu.info() << "failed to register mem " << (*it)->me << " (" << base <<
          //   " + " << (*it)->size << ") : "
          //       << ret;
          //   continue;
          // }

          registered_host_ptrs.push_back(base);

          // now go through each GPU and verify that it got a GPU pointer (it may not
          // match the CPU
          //  pointer, but that's ok because we'll never refer to it directly)
          for(unsigned i = 0; i < dpus.size(); i++) {
            log_dpu.info() << "memory " << (*it)->me
                           << " successfully registered with GPU " << dpus[i]->proc->me;
            dpus[i]->pinned_sysmems.insert((*it)->me);

            // char *gpuptr;
            // hipError_t ret;
            // {
            //   AutoGPUContext agc(dpus[i]);
            //   ret = hipHostGetDevicePointer((void **)&gpuptr, base, 0);
            // }
            // if(ret == hipSuccess) {
            //   // no test for && ((void *)gpuptr == base)) {
            //   log_dpu.info() << "memory " << (*it)->me << " successfully registered
            //   with GPU " << dpus[i]->proc->me;
            //   dpus[i]->pinned_sysmems.insert((*it)->me);
            // } else {
            //   log_dpu.warning() << "GPU #" << i << " has no mapping for registered
            //   memory (" << (*it)->me << " at " << base << ") !?";
            // }
          }
        }
      }

      // ask any ipc-able nodes to share handles with us
      if(config->cfg_use_upmem_ipc) {
        NodeSet ipc_peers = Network::all_peers;

        // #ifdef REALM_ON_LINUX
        //         if(!ipc_peers.empty()) {
        //           log_upmemipc.info() << "requesting upmem ipc handles from "
        //                              << ipc_peers.size() << " peers";

        //           // we'll need a reponse (and ultimately, a release) from each peer
        //           upmemipc_responses_needed.fetch_add(ipc_peers.size());
        //           upmemipc_releases_needed.fetch_add(ipc_peers.size());

        //           ActiveMessage<UpmemIpcRequest> amsg(ipc_peers);
        //           amsg->hostid = gethostid();
        //           amsg.commit();

        //           // wait for responses
        //           {
        //             AutoLock<> al(upmemipc_mutex);
        //             while(upmemipc_responses_needed.load_acquire() > 0)
        //               upmemipc_condvar.wait();
        //           }
        //           log_upmemipc.info() << "responses complete";
        //         }
        // #endif
      }

      for(std::vector<DPU *>::iterator it = dpus.begin(); it != dpus.end(); it++) {
        (*it)->create_dma_channels(runtime);
      }

      Module::create_dma_channels(runtime);
    }

    // create any code translators provided by the module (default == do nothing)
    void UpmemModule::create_code_translators(RuntimeImpl *runtime)
    {
      Module::create_code_translators(runtime);
    }

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void UpmemModule::cleanup(void)
    {
      Module::cleanup();
      size_t dpu_count = 0;
      for(size_t i = config->cfg_skip_dpu_count;
          (i < dpu_info.size()) && (static_cast<int>(dpu_count) < config->cfg_num_dpus);
          i++) {
        delete dpus[++dpu_count];
      }
    }

  }; // namespace Upmem

}; // namespace Realm
