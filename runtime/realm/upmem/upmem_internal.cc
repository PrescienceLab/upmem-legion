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

#include "realm/upmem/upmem_internal.h"

namespace Realm {

  extern Logger log_xd;
  extern Logger log_taskreg;

  namespace Upmem {
    extern Logger log_dpu;
    extern Logger log_stream;
    extern Logger log_dpudma;

    ////////////////////////////////////////////////////////////////////////
    //
    // class UpmemDeviceMemoryInfo

    UpmemDeviceMemoryInfo::UpmemDeviceMemoryInfo(int _device_id)
      : device_id(_device_id)
      , dpu(0)
    {
      // see if we can match this context to one of our DPU objects - handle
      //  the case where the hip module didn't load though
      UpmemModule *mod = get_runtime()->get_module<UpmemModule>("upmem");
      if(mod) {
        for(std::vector<DPU *>::const_iterator it = mod->dpus.begin();
            it != mod->dpus.end(); ++it)
          if((*it)->device_id == _device_id) {
            dpu = *it;
            break;
          }
      }
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPU

    DPU::DPU(UpmemModule *_module, DPUInfo *_info, DPUWorker *_worker, int _device_id)
      : module(_module)
      , info(_info)
      , worker(_worker)
      , proc(0)
      , device_id(_device_id)
    {
      event_pool.init_pool();

      stream = new DPUStream(this, worker);

      struct dpu_set_t single_dpu;

      DPU_ASSERT(dpu_alloc(1, "backend=simulator", &single_dpu));
      printf("DPU ALLOCATED with id %d\n", _device_id);
      stream->set_stream(&single_dpu);

      task_streams.resize(module->config->cfg_task_streams);
      for(unsigned i = 0; i < module->config->cfg_task_streams; i++)
        task_streams[i] = new DPUStream(this, worker);
    }

    DPU::~DPU(void)
    {
      event_pool.empty_pool();

      DPU_ASSERT(dpu_free(*stream->get_stream()));
    }

    void DPU::create_mram_memory(RuntimeImpl *runtime, size_t size)
    {
      // TODO: look into page offset (4096)
      mram_base = (char *)0x8; // physical addressing but realm breaks at 0x0
      Memory m = runtime->next_local_memory_id();
      mram = new DPUMRAMMemory(m, this, stream, static_cast<char *>(mram_base), size);
      runtime->add_memory(mram);
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
    const DPU::UpmemIpcMapping *DPU::find_ipc_mapping(Memory mem) const
    {
      for(std::vector<UpmemIpcMapping>::const_iterator it = upmemipc_mappings.begin();
          it != upmemipc_mappings.end(); ++it)
        if(it->mem == mem)
          return &*it;

      return 0;
    }

    void DPU::create_dma_channels(RuntimeImpl *r)
    {
      r->add_dma_channel(new DPUChannel(this, XFER_DPU_IN_MRAM, &r->bgwork));
      r->add_dma_channel(new DPUfillChannel(this, &r->bgwork));
      // r->add_dma_channel(new DPUreduceChannel(this, &r->bgwork));

      if(!pinned_sysmems.empty()) {
        r->add_dma_channel(new DPUChannel(this, XFER_DPU_TO_MRAM, &r->bgwork));
        r->add_dma_channel(new DPUChannel(this, XFER_DPU_FROM_MRAM, &r->bgwork));
      } else {
        log_dpu.warning() << "DPU " << proc->me << " has no accessible system memories!?";
      }
      // only create a p2p channel if we have peers (and an mram)
      if(!peer_mram.empty() || !upmemipc_mappings.empty()) {
        r->add_dma_channel(new DPUChannel(this, XFER_DPU_PEER_MRAM, &r->bgwork));
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

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUReplHeapListener
    //

    DPUReplHeapListener::DPUReplHeapListener(UpmemModule *_module)
      : module(_module)
    {}

    void DPUReplHeapListener::chunk_created(void *base, size_t bytes)
    {
      if(!module->dpus.empty()) {
        log_dpu.info() << "registering replicated heap chunk: base=" << base
                       << " size=" << bytes;

        base = malloc(bytes);
        if(base == NULL) {
          log_dpu.fatal() << "failed to register replicated heap chunk: base=" << base
                          << " size=" << bytes;
          abort();
        }
      }
    }

    void DPUReplHeapListener::chunk_destroyed(void *base, size_t bytes)
    {
      if(!module->dpus.empty()) {
        log_dpu.info() << "unregistering replicated heap chunk: base=" << base
                       << " size=" << bytes;
        free(base);
      }
    }

  }; // namespace Upmem
};   // namespace Realm