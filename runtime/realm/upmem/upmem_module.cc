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
      
      for(int i = 0; i < (config->cfg_num_dpus / 64); i++) {
          CHECK_UPMEM(dpu_free(*allocated_dpu_sets[i]));
      }
      
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

      if(config->cfg_num_dpus >  MAX_DPUS_PER_RANK) {
        log_dpu.fatal() << "Specified > MAX_DPUS_PER_RANK. Check command line options -ll:num_dpus";
        assert(false);
      }

      // if we are using a shared worker, create that next
      if(config->cfg_use_shared_worker) {
        shared_worker = new DPUWorker;

        if(config->cfg_use_worker_threads)
          shared_worker->start_background_thread(
              runtime->core_reservation_set(),
              config->cfg_mram_mem_size);
        else
          shared_worker->add_to_manager(&(runtime->bgwork));
      }

      // each DPU has 64MB of MRAM and 64KB of cache
      // we need to add each DPU to the core reservation in Realm

      dpus.resize(config->cfg_num_dpus);
      dpu_info.resize(config->cfg_num_dpus);

      uint64_t dpu_count = 0;
      // try to get cfg_num_dpus, working through the list in order
      while(dpu_count < (uint64_t)config->cfg_num_dpus) { 
  
        uint64_t dpus_allocated = std::min<uint64_t>(config->cfg_num_dpus, 64);

        dpu_set_t *allocated_dpu_set_single = new dpu_set_t;
        allocated_dpu_sets.push_back(allocated_dpu_set_single);

  #if !defined(__SIMULATOR__)
        CHECK_UPMEM(dpu_alloc(dpus_allocated, "backend=hw", allocated_dpu_set_single));
  #else
        CHECK_UPMEM(dpu_alloc(dpus_allocated, "backend=simulator", allocated_dpu_set_single));
  #endif

        for(uint64_t j=0; j < dpus_allocated; j++) {
          DPUWorker *worker;
          if(config->cfg_use_shared_worker) {
            worker = shared_worker;
          } else {
            worker = new DPUWorker;

            if(config->cfg_use_worker_threads)
              worker->start_background_thread(runtime->core_reservation_set(),
                                              config->cfg_mram_mem_size);
            else
              worker->add_to_manager(&(runtime->bgwork));
          }
          DPU *g = new DPU(this, dpu_info[dpu_count], worker, dpu_count);

          if(!config->cfg_use_shared_worker)
            dedicated_workers[g] = worker;

          dpus[dpu_count++] = g;
        }
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
      // a single ZC memory for everybody
      if((config->cfg_zc_mem_size > 0) && !dpus.empty()) {
        char *zcmem_dpu_base;
        {
          zcmem_dpu_base = (char *)malloc(config->cfg_zc_ib_size);
        }
        if(zcmem_dpu_base == NULL) {
          log_dpu.fatal() << "insufficient device-mappable host memory: "
                          << config->cfg_zc_mem_size << " bytes needed (from -ll:zsize)";

          abort();
        }

        Memory m = runtime->next_local_memory_id();
        zcmem = new DPUZCMemory(m, zcmem_dpu_base, config->cfg_zc_mem_size);
        runtime->add_memory(zcmem);

        // add the ZC memory as a pinned memory to all DPUs
        for(unsigned i = 0; i < dpus.size(); i++) {
          dpus[i]->pinned_sysmems.insert(zcmem->me);
        }
      }

      // allocate intermediate buffers in ZC memory for DMA engine
      if((config->cfg_zc_ib_size > 0) && !dpus.empty()) {
        char *zcib_cpu_base;
        {
          zcib_cpu_base = (char *)malloc(config->cfg_zc_ib_size);
        }
        Memory m = runtime->next_local_ib_memory_id();
        IBMemory *ib_mem;
        ib_mem = new IBMemory(m, config->cfg_zc_ib_size, MemoryImpl::MKIND_ZEROCOPY,
                              Memory::Z_COPY_MEM, zcib_cpu_base, 0);
        runtime->add_ib_memory(ib_mem);

        // add the IB memory as a pinned memory to all the DPUs
        for(unsigned i = 0; i < dpus.size(); i++) {
          dpus[i]->pinned_sysmems.insert(ib_mem->me);
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
        (*it)->create_processor(runtime, config->cfg_mram_mem_size);
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
            assert(0 && "We should not be here\n");
            continue;
          }

          void *base = (*it)->get_direct_ptr(0, (*it)->size);

          if(base == 0)
            continue;

          registered_host_ptrs.push_back(base);

          // now go through each DPU
          for(unsigned i = 0; i < dpus.size(); i++) {
            log_dpu.info() << "memory " << (*it)->me
                           << " successfully registered with DPU " << dpus[i]->proc->me;
            dpus[i]->pinned_sysmems.insert((*it)->me);
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
      // clean up worker(s)
      if(shared_worker) {
#ifdef DEBUG_REALM
        shared_worker->shutdown_work_item();
#endif
        if(config->cfg_use_worker_threads)
          shared_worker->shutdown_background_thread();

        delete shared_worker;
        shared_worker = 0;
      }

      for(std::map<DPU *, DPUWorker *>::iterator it = dedicated_workers.begin();
          it != dedicated_workers.end(); it++) {
        DPUWorker *worker = it->second;

#ifdef DEBUG_REALM
        worker->shutdown_work_item();
#endif
        if(config->cfg_use_worker_threads)
          worker->shutdown_background_thread();

        delete worker;
      }
      dedicated_workers.clear();

      // also unregister any host memory at this time
      if(!registered_host_ptrs.empty()) {
        registered_host_ptrs.clear();
      }

      // and clean up anything that was needed for the replicated heap
      runtime->repl_heap.remove_listener(rh_listener);

      size_t dpu_count = 0;
      for(size_t i = config->cfg_skip_dpu_count;
          (i < dpu_info.size()) && (static_cast<int>(dpu_count) < config->cfg_num_dpus);
          i++, dpu_count++) {
        delete dpus[dpu_count];
      }

      dpus.clear();

      Module::cleanup();
    }

  }; // namespace Upmem

}; // namespace Realm
