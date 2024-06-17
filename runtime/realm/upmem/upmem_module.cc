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
#include "realm/upmem/upmem_internal.h"
#include "realm/upmem/upmem_access.h"

#include "realm/tasks.h"
#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/event_impl.h"
#include "realm/idx_impl.h"

#include "realm/transfer/lowlevel_dma.h"
#include "realm/transfer/channel.h"
#include "realm/transfer/ib_memory.h"

#include "realm/mutex.h"
#include "realm/utils.h"

#include <stdio.h>
#include <string.h>
#include <dlfcn.h>

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

    void DPU::create_mram_memory(RuntimeImpl *runtime, size_t size)
    {
      Memory m = runtime->next_local_memory_id();
      mram = new DPUMRAMMemory(m, this, stream, static_cast<char *>(mram_base), size);
      runtime->add_memory(mram);
    }

    DPUMRAMMemory::DPUMRAMMemory(Memory _me, DPU *_dpu, DPUStream *_stream, char *_base,
                                 size_t _size)
      : LocalManagedMemory(_me, _size, MKIND_MRAM, 512, Memory::DPU_MRAM_MEM, 0)
      , dpu(_dpu)
      , base(_base)
      , stream(_stream)
    {}

    DPUMRAMMemory::~DPUMRAMMemory(void) {}

    // these work, but they are SLOW
    void DPUMRAMMemory::get_bytes(off_t offset, void *dst, size_t size)
    {
      // use a blocking copy - host memory probably isn't pinned anyway
      {
        // we need to get the dpu_set_t stream
        CHECK_UPMEM(dpu_copy_from(*stream->get_stream(), "data", offset, dst, size));
      }
    }

    void DPUMRAMMemory::put_bytes(off_t offset, const void *src, size_t size)
    {
      // use a blocking copy - host memory probably isn't pinned anyway
      {
        // we need to get the dpu_set_t stream
        CHECK_UPMEM(dpu_broadcast_to(*stream->get_stream(), "data", offset, src, size,
                                     DPU_XFER_DEFAULT));
      }
    }

    void *DPUMRAMMemory::get_direct_ptr(off_t offset, size_t size)
    {
      return (void *)(base + offset);
    }

    void DPU::create_processor(RuntimeImpl *runtime, size_t stack_size)
    {
      Processor p = runtime->next_local_processor_id();
      proc = new DPUProcessor(this, p, runtime->core_reservation_set(), stack_size);
      runtime->add_processor(proc);

      // this processor is able to access its own FB and the ZC mem (if any)
      Machine::ProcessorMemoryAffinity pma;
      pma.p = p;
      pma.m = mram->me;
      // pma.bandwidth = info->logical_peer_bandwidth[info->index];
      // pma.latency   = info->logical_peer_latency[info->index];
      runtime->add_proc_mem_affinity(pma);
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
      Module::create_dma_channels(runtime);
    }

    // create any code translators provided by the module (default == do nothing)
    void UpmemModule::create_code_translators(RuntimeImpl *runtime)
    {
      Module::create_code_translators(runtime);
    }

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void UpmemModule::cleanup(void) { Module::cleanup(); }

    ////////////////////////////////////////////////////////////////////////
    //
    // class ContextSynchronizer

    ContextSynchronizer::ContextSynchronizer(DPU *_dpu, int _device_id,
                                             CoreReservationSet &crs, int _max_threads)
      : dpu(_dpu)
      , device_id(_device_id)
      , max_threads(_max_threads)
      , condvar(mutex)
      , shutdown_flag(false)
      , total_threads(0)
      , sleeping_threads(0)
      , syncing_threads(0)
    {
      Realm::CoreReservationParameters params;
      params.set_num_cores(1);
      params.set_alu_usage(params.CORE_USAGE_SHARED);
      params.set_fpu_usage(params.CORE_USAGE_MINIMAL);
      params.set_ldst_usage(params.CORE_USAGE_MINIMAL);
      params.set_max_stack_size(64 * MEGABYTE);

      std::string name = stringbuilder() << "DPU ctxsync " << device_id;

      core_rsrv = new Realm::CoreReservation(name, crs, params);
    }

    ContextSynchronizer::~ContextSynchronizer()
    {
      assert(total_threads == 0);
      delete core_rsrv;
    }

    void ContextSynchronizer::shutdown_threads()
    {
      // set the shutdown flag and wake up everybody
      {
        AutoLock<> al(mutex);
        shutdown_flag = true;
        if(sleeping_threads > 0)
          condvar.broadcast();
      }

      for(int i = 0; i < total_threads; i++) {
        worker_threads[i]->join();
        delete worker_threads[i];
      }

      worker_threads.clear();
      total_threads = false;
      sleeping_threads = false;
      syncing_threads = false;
      shutdown_flag = false;
    }

    void ContextSynchronizer::add_fence(DPUWorkFence *fence)
    {
      bool start_new_thread = false;
      {
        AutoLock<> al(mutex);

        fences.push_back(fence);

        // if all the current threads are asleep or busy syncing, we
        //  need to do something
        if((sleeping_threads + syncing_threads) == total_threads) {
          // is there a sleeping thread we can wake up to handle this?
          if(sleeping_threads > 0) {
            // just poke one of them
            condvar.signal();
          } else {
            // can we start a new thread?  (if not, we'll just have to
            //  be patient)
            if(total_threads < max_threads) {
              total_threads++;
              syncing_threads++; // threads starts as if it's syncing
              start_new_thread = true;
            }
          }
        }
      }

      if(start_new_thread) {
        Realm::ThreadLaunchParameters tlp;

        Thread *t =
            Realm::Thread::create_kernel_thread<ContextSynchronizer,
                                                &ContextSynchronizer::thread_main>(
                this, tlp, *core_rsrv, 0);
        // need the mutex to put this thread in the list
        {
          AutoLock<> al(mutex);
          worker_threads.push_back(t);
        }
      }
    }

    void ContextSynchronizer::thread_main()
    {
      while(true) {
        DPUWorkFence::FenceList my_fences;

        // attempt to get a non-empty list of fences to synchronize,
        //  sleeping when needed and paying attention to the shutdown
        //  flag
        {
          AutoLock<> al(mutex);

          syncing_threads--;

          while(true) {
            if(shutdown_flag)
              return;

            if(fences.empty()) {
              // sleep until somebody tells us there's stuff to do
              sleeping_threads++;
              condvar.wait();
              sleeping_threads--;
            } else {
              // grab everything (a single sync covers however much stuff
              //  was pushed ahead of it)
              syncing_threads++;
              my_fences.swap(fences);
              break;
            }
          }
        }

        // shouldn't get here with an empty list
        assert(!my_fences.empty());

        // mark all the fences complete
        while(!my_fences.empty()) {
          DPUWorkFence *fence = my_fences.pop_front();
          fence->mark_finished(true /*successful*/);
        }

        // and go back around for more...
      }
    }

    DPUStream *DPU::find_stream(struct dpu_set_t *stream) const
    {
      for(std::vector<DPUStream *>::const_iterator it = task_streams.begin();
          it != task_streams.end(); it++)
        if((*it)->get_stream() == stream)
          return *it;
      return NULL;
    }

    DPUStream *DPU::get_null_task_stream(void) const
    {
      DPUStream *stream = ThreadLocal::current_dpu_stream;
      assert(stream != NULL);
      return stream;
    }

    DPUStream *DPU::get_next_task_stream(bool create)
    {
      if(create && !ThreadLocal::created_dpu_streams) {
        // First time we get asked to create, user our current stream
        ThreadLocal::created_dpu_streams = new std::set<DPUStream *>();
        assert(ThreadLocal::current_dpu_stream);
        ThreadLocal::created_dpu_streams->insert(ThreadLocal::current_dpu_stream);
        return ThreadLocal::current_dpu_stream;
      }
      unsigned index = next_task_stream.fetch_add(1) % task_streams.size();
      DPUStream *result = task_streams[index];
      if(create)
        ThreadLocal::created_dpu_streams->insert(result);
      return result;
    }

    struct dpu_set_t *UpmemModule::get_task_upmem_stream()
    {
      // if we're not in a dpu task, this'll be null
      if(ThreadLocal::current_dpu_stream)
        return ThreadLocal::current_dpu_stream->get_stream();
      else
        return 0;
    }

  }; // namespace Upmem

}; // namespace Realm
