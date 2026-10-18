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

    namespace ThreadLocal {
      static REALM_THREAD_LOCAL DPUProcessor *current_dpu_proc = 0;
      static REALM_THREAD_LOCAL DPUStream *current_dpu_stream = 0;
      static REALM_THREAD_LOCAL std::set<DPUStream *> *created_dpu_streams = 0;
      static REALM_THREAD_LOCAL int context_sync_required = 0;
    }; // namespace ThreadLocal

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

    struct dpu_set_t *UpmemModule::get_task_upmem_stream()
    {
      // if we're not in a dpu task, this'll be null
      if(ThreadLocal::current_dpu_stream)
        return ThreadLocal::current_dpu_stream->get_stream();
      else
        return 0;
    }

    DPU *UpmemModule::get_task_dpu()
    {
      if(ThreadLocal::current_dpu_proc)
        return ThreadLocal::current_dpu_proc->dpu;
      else
        return 0;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPU
    DPU::DPU(UpmemModule *_module, DPUWorker *_worker, int _device_id,
             dpu_set_t *_rank_set)
      : module(_module)
      , info(0)
      , worker(_worker)
      , proc(0)
      , mram(0)
      , device_id(_device_id)
      , granule(_module->config->cfg_dpu_granule)
    {
      event_pool.init_pool();

      assert(module->config->cfg_task_streams > 0 &&
             "cfg_task_streams should be set to zero");

      CHECK_UPMEM(dpu_get_nr_dpus(*_rank_set, &nr_dpus));
      assert(nr_dpus > 0 && nr_dpus <= MAX_DPUS_PER_RANK);
      dpu_set_t dpu;
      DPU_FOREACH(*_rank_set, dpu) { dpu_handles.push_back(dpu); }
      assert(dpu_handles.size() == nr_dpus);

      // fixed MRAM offset, independent of the loaded binary's heap pointer
      mram_data.address = 0x08000000 /* host-side MRAM address mask */ + REALM_UPMEM_MRAM_DATA_OFFSET;
      mram_data.size = REALM_UPMEM_MRAM_DATA_SIZE;

      log_dpu.info() << "rank " << device_id << ": " << nr_dpus
                     << " DPUs, granule=" << granule;

      stream = new DPUStream(this, worker);
      stream->set_stream(_rank_set);
    }

    DPU::~DPU(void)
    {
      event_pool.empty_pool();
    }

    void DPU::create_mram_memory(RuntimeImpl *runtime, size_t size)
    {
      mram_base = reinterpret_cast<char *>(uintptr_t(REALM_UPMEM_RANK_PTR_BASE));
      // whole rows only
      size_t rank_size = (size / granule) * granule * nr_dpus;
      Memory m = runtime->next_local_memory_id();
      mram = new DPUMRAMMemory(m, this, stream, mram_base, rank_size);
      runtime->add_memory(mram);
    }

    void DPU::create_processor(RuntimeImpl *runtime, size_t stack_size)
    {
      Processor p = runtime->next_local_processor_id();
      proc = new DPUProcessor(this, p, runtime->core_reservation_set(), stack_size);
      runtime->add_processor(proc);

      if(mram) {
        Machine::ProcessorMemoryAffinity pma;
        pma.p = p;
        pma.m = mram->me;
        pma.bandwidth = 200;
        pma.latency = 5;
        runtime->add_proc_mem_affinity(pma);
      }

      for(std::set<Memory>::const_iterator it = pinned_sysmems.begin();
          it != pinned_sysmems.end(); ++it) {
        // no processor affinity to IB memories
        if(!ID(*it).is_memory())
          continue;

        Machine::ProcessorMemoryAffinity pma;
        pma.p = p;
        pma.m = *it;
        pma.bandwidth = 20; // "medium"
        pma.latency = 200;  // "bad"
        runtime->add_proc_mem_affinity(pma);
      }

      // no peer access between ranks

      // look for any other local memories that belong to our context or
      //  peer-able contexts
      const Node &n = get_runtime()->nodes[Network::my_node_id];
      for(std::vector<MemoryImpl *>::const_iterator it = n.memories.begin();
          it != n.memories.end(); ++it) {
        UpmemDeviceMemoryInfo *cdm = (*it)->find_module_specific<UpmemDeviceMemoryInfo>();
        if(!cdm)
          continue;
        if(cdm->device_id == device_id) {
          Machine::ProcessorMemoryAffinity pma;
          pma.p = p;
          pma.m = (*it)->me;
          pma.bandwidth = 200; // "big"
          pma.latency = 5;     // "ok"
          runtime->add_proc_mem_affinity(pma);
        } else {
          if(cdm->dpu && (info->peers.count(cdm->dpu->info->device) > 0)) {
            Machine::ProcessorMemoryAffinity pma;
            pma.p = p;
            pma.m = (*it)->me;
            pma.bandwidth = 10; // assuming pcie, this should be ~half the bw and
            pma.latency = 400;  // ~twice the latency as zcmem
            runtime->add_proc_mem_affinity(pma);
          }
        }
      }
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
      if(!mram)
	      return;

      // no in-MRAM copy channel; reduce channel disabled (not layout-aware)
      r->add_dma_channel(new DPUfillChannel(this, &r->bgwork));

      // zero copy host memory is registered in pinned_sysmems. This enables a path from 
      // host to UPMEM DPU and vice versa
      if(!pinned_sysmems.empty()) {
        r->add_dma_channel(new DPUChannel(this, XFER_DPU_TO_MRAM, &r->bgwork));
        r->add_dma_channel(new DPUChannel(this, XFER_DPU_FROM_MRAM, &r->bgwork));
      } else {
        log_dpu.warning() << "DPU " << proc->me << " has no accessible system memories!?";
      }
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // rank memory copies
    //
    // split: partial words (RMW), partial rows (<=3 pushes), full rows (SG push)

    namespace {
      struct RowGatherArgs {
        uint8_t *host;
        uint64_t row_bytes;
        uint32_t granule;
        uint32_t rows;
      };

      bool row_gather_block(struct sg_block_info *out, uint32_t dpu_index,
                            uint32_t block_index, void *_args)
      {
        const RowGatherArgs *args = static_cast<const RowGatherArgs *>(_args);
        if(block_index >= args->rows)
          return false;
        out->addr = args->host + (block_index * args->row_bytes) +
                    (uint64_t(dpu_index) * args->granule);
        out->length = args->granule;
        return true;
      }
    }; // namespace

    void DPU::enqueue_copy(DPUMemcpyKind kind, void *host, uint64_t rank_addr,
                           size_t bytes)
    {
      if(bytes == 0)
        return;

      uint8_t *h = static_cast<uint8_t *>(host);
      uint64_t end = rank_addr + bytes;
      uint64_t word_lo = (rank_addr + 7) & ~uint64_t(7);
      uint64_t word_hi = end & ~uint64_t(7);

      if(word_lo >= word_hi) {
        // no whole word - at most two partial ones
        uint64_t split = std::min(end, word_lo);
        copy_partial_word(kind, h, rank_addr, split - rank_addr);
        if(split < end)
          copy_partial_word(kind, h + (split - rank_addr), split, end - split);
        return;
      }

      if(rank_addr < word_lo)
        copy_partial_word(kind, h, rank_addr, word_lo - rank_addr);
      enqueue_aligned_copy(kind, h + (word_lo - rank_addr), word_lo, word_hi - word_lo);
      if(word_hi < end)
        copy_partial_word(kind, h + (word_hi - rank_addr), word_hi, end - word_hi);
    }

    void DPU::copy_sync(DPUMemcpyKind kind, void *host, uint64_t rank_addr, size_t bytes)
    {
      enqueue_copy(kind, host, rank_addr, bytes);
      AutoLock<> al(xfer_mutex);
      CHECK_UPMEM(dpu_sync(*stream->get_stream()));
    }

    void DPU::copy_partial_word(DPUMemcpyKind kind, uint8_t *host, uint64_t rank_addr,
                                size_t bytes)
    {
      assert((rank_addr & ~uint64_t(7)) == ((rank_addr + bytes - 1) & ~uint64_t(7)));
      uint32_t d = realm_upmem_owner(rank_addr, nr_dpus, granule);
      uint64_t local = realm_upmem_local(rank_addr, nr_dpus, granule);
      uint64_t word = local & ~uint64_t(7);
      unsigned shift = local - word;

      AutoLock<> al(xfer_mutex);
      // sync calls queued behind async work with pending callbacks hang or crash
      //  the UPMEM runtime (2025.1), so drain the rank first
      CHECK_UPMEM(dpu_sync(*stream->get_stream()));
      uint8_t buffer[8];
      CHECK_UPMEM(dpu_copy_from_symbol(dpu_handles[d], mram_data, word, buffer, 8));
      if(kind == DPU_XFER_TO_DPU) {
        memcpy(buffer + shift, host, bytes);
        CHECK_UPMEM(dpu_copy_to_symbol(dpu_handles[d], mram_data, word, buffer, 8));
      } else {
        memcpy(host, buffer + shift, bytes);
      }
    }

    void DPU::enqueue_aligned_copy(DPUMemcpyKind kind, uint8_t *host, uint64_t rank_addr,
                                   size_t bytes)
    {
      const uint64_t row = row_bytes();
      uint64_t v = rank_addr;
      const uint64_t end = rank_addr + bytes;

      // leading partial row
      if((v % row) != 0) {
        uint64_t len = std::min(end, (v / row + 1) * row) - v;
        enqueue_row_copy(kind, host, v, len);
        host += len;
        v += len;
      }

      // full rows
      while((end - v) >= row) {
        uint64_t rows = std::min<uint64_t>((end - v) / row,
                                           module->config->cfg_sg_max_blocks);
        if(rows == 1) {
          enqueue_row_copy(kind, host, v, row);
        } else {
          RowGatherArgs args;
          args.host = host;
          args.row_bytes = row;
          args.granule = granule;
          args.rows = rows;
          get_block_t get_block;
          get_block.f = &row_gather_block;
          get_block.args = &args; // copied by the SDK
          get_block.args_size = sizeof(args);

          AutoLock<> al(xfer_mutex);
          CHECK_UPMEM(dpu_push_sg_xfer_symbol(*stream->get_stream(), kind, mram_data,
                                              (v / row) * granule, rows * granule,
                                              &get_block, DPU_SG_XFER_ASYNC));
        }
        host += rows * row;
        v += rows * row;
      }

      // trailing partial row
      if(v < end)
        enqueue_row_copy(kind, host, v, end - v);
    }

    void DPU::enqueue_row_copy(DPUMemcpyKind kind, uint8_t *host, uint64_t rank_addr,
                               size_t bytes)
    {
      // group pieces by (local offset, length)
      std::map<std::pair<uint64_t, uint64_t>, std::vector<void *>> groups;
      uint64_t v = rank_addr;
      const uint64_t end = rank_addr + bytes;
      while(v < end) {
        uint64_t piece = std::min(end, (v / granule + 1) * granule) - v;
        uint32_t d = realm_upmem_owner(v, nr_dpus, granule);
        std::vector<void *> &ptrs =
            groups[std::make_pair(realm_upmem_local(v, nr_dpus, granule), piece)];
        if(ptrs.empty())
          ptrs.resize(nr_dpus, 0);
        ptrs[d] = host + (v - rank_addr);
        v += piece;
      }

      for(std::map<std::pair<uint64_t, uint64_t>, std::vector<void *>>::const_iterator
              it = groups.begin();
          it != groups.end(); ++it)
        push_dpus(kind, it->second, it->first.first, it->first.second);
    }

    void DPU::push_dpus(DPUMemcpyKind kind, const std::vector<void *> &ptrs,
                        uint64_t local, size_t bytes)
    {
      // the SDK doesn't clear prepared buffers after a push, so set all of them
      AutoLock<> al(xfer_mutex);
      for(uint32_t i = 0; i < nr_dpus; i++)
        CHECK_UPMEM(dpu_prepare_xfer(dpu_handles[i], ptrs[i]));
      CHECK_UPMEM(dpu_push_xfer_symbol(*stream->get_stream(), kind, mram_data, local,
                                       bytes, DPU_XFER_ASYNC));
    }

    DPUStream *DPU::get_null_stream(void) const
    {
      DPUStream *stream = ThreadLocal::current_dpu_stream;
      assert(stream != NULL);
      return stream;
    }

    DPUStream *DPU::get_next_stream(bool create)
    {
      // if(create && !ThreadLocal::created_dpu_streams) {
      //   // First time we get asked to create, user our current stream
      //   ThreadLocal::created_dpu_streams = new std::set<DPUStream *>();
      //   assert(ThreadLocal::current_dpu_stream);
      //   ThreadLocal::created_dpu_streams->insert(ThreadLocal::current_dpu_stream);
      //   return ThreadLocal::current_dpu_stream;
      // }
      // // unsigned index = next_task_stream.fetch_add(1) % task_streams.size();
      // DPUStream *result = task_streams[0];
      // if(create)
      //   ThreadLocal::created_dpu_streams->insert(result);
      // return result;

      assert(stream != NULL);
      return stream;
    }

    ///////////////////////////////////////////////////////////////////////
    //
    // class DPUProcessor

    DPUProcessor::DPUProcessor(DPU *_dpu, Processor _me, Realm::CoreReservationSet &crs,
                               size_t _stack_size)
      : LocalTaskProcessor(_me, Processor::DPU_PROC)
      , dpu(_dpu)
      , block_on_synchronize(false)
      , ctxsync(_dpu, _dpu->device_id, crs, _dpu->module->config->cfg_max_ctxsync_threads)
    {
      // shared: more ranks (40) than cores
      Realm::CoreReservationParameters params;
      params.set_num_cores(1);
      params.set_alu_usage(params.CORE_USAGE_SHARED);
      params.set_fpu_usage(params.CORE_USAGE_SHARED);
      params.set_ldst_usage(params.CORE_USAGE_SHARED);
      params.set_max_stack_size(_stack_size);

      std::string name = stringbuilder() << "DPU proc " << _me;

      // Each rank gets a new Realm core reservation
      core_rsrv = new Realm::CoreReservation(name, crs, params);

      Realm::KernelThreadTaskScheduler *sched =
          new DPUTaskScheduler<Realm::KernelThreadTaskScheduler>(me, *core_rsrv, this);

      set_scheduler(sched);
    }

    DPUProcessor::~DPUProcessor(void) { delete core_rsrv; }

    bool DPUProcessor::register_task(Processor::TaskFuncID func_id,
                                     CodeDescriptor &codedesc,
                                     const ByteArrayRef &user_data)
    {
      // see if we have a function pointer to register
      const FunctionPointerImplementation *fpi =
          codedesc.find_impl<FunctionPointerImplementation>();

      assert(fpi != 0);

      {
        RWLock::AutoWriterLock al(task_table_mutex);

        // first, make sure we haven't seen this task id before
        if(dpu_task_table.count(func_id) > 0) {
          log_taskreg.fatal() << "duplicate task registration: proc=" << me
                              << " func=" << func_id;
          return false;
        }

        DPUTaskTableEntry &tte = dpu_task_table[func_id];

        // figure out what type of function we have
        if(codedesc.type() == TypeConv::from_cpp_type<Processor::TaskFuncPtr>()) {
          tte.fnptr = (Processor::TaskFuncPtr)(fpi->fnptr);
          tte.stream_aware_fnptr = 0;
        } else if(codedesc.type() ==
                  TypeConv::from_cpp_type<Upmem::StreamAwareTaskFuncPtr>()) {
          tte.fnptr = 0;
          tte.stream_aware_fnptr = (Upmem::StreamAwareTaskFuncPtr)(fpi->fnptr);
        } else {
          log_taskreg.fatal() << "attempt to register a task function of improper type: "
                              << codedesc.type();
          assert(0);
        }

        tte.user_data = user_data;
      }

      log_taskreg.info() << "task " << func_id << " registered on " << me << ": "
                         << codedesc;

      return true;
    }

    void DPUProcessor::execute_task(Processor::TaskFuncID func_id,
                                    const ByteArrayRef &task_args)
    {
      if(func_id == Processor::TASK_ID_PROCESSOR_NOP)
        return;

      std::map<Processor::TaskFuncID, DPUTaskTableEntry>::const_iterator it;
      {
        RWLock::AutoReaderLock al(task_table_mutex);

        it = dpu_task_table.find(func_id);
        if(it == dpu_task_table.end()) {
          log_taskreg.fatal() << "task " << func_id << " not registered on " << me;
          assert(0);
        }
      }

      const DPUTaskTableEntry &tte = it->second;

      if(tte.stream_aware_fnptr) {
        // shouldn't be here without a valid stream
        assert(ThreadLocal::current_dpu_stream);
        struct dpu_set_t *stream = ThreadLocal::current_dpu_stream->get_stream();

        log_taskreg.debug() << "task " << func_id << " executing on " << me << ": "
                            << ((void *)(tte.stream_aware_fnptr)) << " (stream aware)";

        (tte.stream_aware_fnptr)(task_args.base(), task_args.size(), tte.user_data.base(),
                                 tte.user_data.size(), me, stream);
      } else {
        assert(tte.fnptr);
        log_taskreg.debug() << "task " << func_id << " executing on " << me << ": "
                            << ((void *)(tte.fnptr));

        (tte.fnptr)(task_args.base(), task_args.size(), tte.user_data.base(),
                    tte.user_data.size(), me);
      }
    }

    void DPUProcessor::shutdown(void)
    {
      log_dpu.info("shutting down");

      // CHECK_UPMEM(dpu_sync(*dpu->stream->get_stream()));

      // shut down threads/scheduler
      LocalTaskProcessor::shutdown();

      ctxsync.shutdown_threads();
    }

    void DPUProcessor::device_synchronize(void)
    {
      DPUStream *current = ThreadLocal::current_dpu_stream;

      if(ThreadLocal::created_dpu_streams) {
        current->wait_on_streams(*ThreadLocal::created_dpu_streams);
        delete ThreadLocal::created_dpu_streams;
        ThreadLocal::created_dpu_streams = 0;
      }

      if(!block_on_synchronize) {
        // We don't actually want to block the GPU processor
        // when synchronizing, so we instead register a cuda
        // event on the stream and then use it triggering to
        // indicate that the stream is caught up
        // Make a completion notification to be notified when
        // the event has actually triggered
        DPUPreemptionWaiter waiter(dpu);
        // Register the waiter with the stream
        current->add_notification(&waiter);
        // Perform the wait, this will preempt the thread
        waiter.preempt();
      } else {
        // oh well...
        CHECK_UPMEM(dpu_sync(*current->get_stream()));
      }
    }

    void DPUProcessor::stream_wait_on_event(dpu_set_t *stream, upmemEvent_t event)
    {
      HERE();
      assert(0 && "should not be here");
      // if (IS_DEFAULT_STREAM(stream))
      //   CHECK_HIP( hipStreamWaitEvent(
      //         ThreadLocal::current_gpu_stream->get_stream(), event, 0) );
      // else
      //   CHECK_HIP( hipStreamWaitEvent(stream, event, 0) );
    }

    void DPUProcessor::stream_synchronize(dpu_set_t *stream)
    {

      if(!block_on_synchronize) {
        DPUStream *s = dpu->get_null_stream();
        if(s) {
          // We don't actually want to block the GPU processor
          // when synchronizing, so we instead register a cuda
          // event on the stream and then use it triggering to
          // indicate that the stream is caught up
          // Make a completion notification to be notified when
          // the event has actually triggered
          DPUPreemptionWaiter waiter(dpu);
          // Register the waiter with the stream
          s->add_notification(&waiter);
          // Perform the wait, this will preempt the thread
          waiter.preempt();
        } else {
          log_dpu.warning() << "WARNING: Detected unknown UPMEM stream " << stream
                            << " that Realm did not create which suggests "
                            << "that there is another copy of the UPMEM runtime "
                            << "somewhere making its own streams... be VERY careful.";
          CHECK_UPMEM(dpu_sync(*ThreadLocal::current_dpu_stream->get_stream()));
        }
      } else {
        // oh well...
        CHECK_UPMEM(dpu_sync(*ThreadLocal::current_dpu_stream->get_stream()));
      }
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUTaskScheduler<T>

    template <typename T>
    DPUTaskScheduler<T>::DPUTaskScheduler(Processor _proc,
                                          Realm::CoreReservation &_core_rsrv,
                                          DPUProcessor *_dpu_proc)
      : T(_proc, _core_rsrv)
      , dpu_proc(_dpu_proc)
    {
      // nothing else
    }

    template <typename T>
    DPUTaskScheduler<T>::~DPUTaskScheduler(void)
    {}

    template <typename T>
    bool DPUTaskScheduler<T>::execute_task(Task *task)
    {
      // use TLS to make sure that the task can find the current DPU processor
      //  UPMEM RT calls
      // TODO: either eliminate these asserts or do TLS swapping when using user threads
      assert(ThreadLocal::current_dpu_proc == 0);
      ThreadLocal::current_dpu_proc = dpu_proc;

      // bump the current stream
      // TODO: sanity-check whether this even works right when DPU tasks suspend
      assert(ThreadLocal::current_dpu_stream == 0);
      DPUStream *s = dpu_proc->dpu->get_next_stream();
      ThreadLocal::current_dpu_stream = s;
      assert(!ThreadLocal::created_dpu_streams);

      // we'll use a "work fence" to track when the kernels launched by this task actually
      //  finish - this must be added to the task _BEFORE_ we execute
      DPUWorkFence *fence = new DPUWorkFence(task);
      task->add_async_work_item(fence);

      // event to record the DPU start time for the task, if requested
      if(task->wants_gpu_work_start()) {
        DPUWorkStart *start = new DPUWorkStart(task);
        task->add_async_work_item(start);
        start->enqueue_on_stream(s);
      }

      bool ok = T::execute_task(task);

      // if the user could have put work on any other streams then make our
      // stream wait on those streams as well
      // TODO: update this so that it works when DPU tasks suspend
      if(ThreadLocal::created_dpu_streams) {
        s->wait_on_streams(*ThreadLocal::created_dpu_streams);
        delete ThreadLocal::created_dpu_streams;
        ThreadLocal::created_dpu_streams = 0;
      }

      // if this is our first task, we might need to decide whether
      //  full context synchronization is required for a task to be
      //  "complete"
      if(dpu_proc->dpu->module->config->cfg_task_context_sync < 0) {
        // without hijack or legacy sync requested, ctxsync is needed
        dpu_proc->dpu->module->config->cfg_task_context_sync = 1;
      }

      if((ThreadLocal::context_sync_required > 0) ||
         ((ThreadLocal::context_sync_required < 0) &&
          dpu_proc->dpu->module->config->cfg_task_context_sync))
        dpu_proc->ctxsync.add_fence(fence);
      else
        fence->enqueue_on_stream(s);

        // A useful debugging macro
#ifdef FORCE_DPU_STREAM_SYNCHRONIZE
      CHECK_UPMEM(dpu_sync(*s->get_stream()));
#endif

      assert(ThreadLocal::current_dpu_proc == dpu_proc);
      ThreadLocal::current_dpu_proc = 0;
      assert(ThreadLocal::current_dpu_stream == s);
      ThreadLocal::current_dpu_stream = 0;

      return ok;
    }

    template <typename T>
    void DPUTaskScheduler<T>::execute_internal_task(InternalTask *task)
    {
      // use TLS to make sure that the task can find the current DPU processor when it
      // makes
      //  UPMEM RT calls
      // TODO: either eliminate these asserts or do TLS swapping when using user threads
      assert(ThreadLocal::current_dpu_proc == 0);
      ThreadLocal::current_dpu_proc = dpu_proc;

      assert(ThreadLocal::current_dpu_stream == 0);
      DPUStream *s = dpu_proc->dpu->get_next_stream();
      ThreadLocal::current_dpu_stream = s;
      assert(!ThreadLocal::created_dpu_streams);

      // internal tasks aren't allowed to wait on events, so any cuda synch
      //  calls inside the call must be blocking
      dpu_proc->block_on_synchronize = true;
      // execute the internal task, whatever it is
      T::execute_internal_task(task);

      // if the user could have put work on any other streams then make our
      // stream wait on those streams as well
      // TODO: update this so that it works when DPU tasks suspend
      if(ThreadLocal::created_dpu_streams) {
        s->wait_on_streams(*ThreadLocal::created_dpu_streams);
        delete ThreadLocal::created_dpu_streams;
        ThreadLocal::created_dpu_streams = 0;
      }

      // we didn't use streams here, so synchronize the whole context
      {
        AutoLock<> al(dpu_proc->dpu->get_xfer_mutex());
        CHECK_UPMEM(dpu_sync(*s->get_stream()));
      }
      dpu_proc->block_on_synchronize = false;

      assert(ThreadLocal::current_dpu_proc == dpu_proc);
      ThreadLocal::current_dpu_proc = 0;
      assert(ThreadLocal::current_dpu_stream == s);
      ThreadLocal::current_dpu_stream = 0;
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
      params.set_fpu_usage(params.CORE_USAGE_SHARED);
      params.set_ldst_usage(params.CORE_USAGE_SHARED);
      params.set_max_stack_size(1 << 20);

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
        // free(base);
      }
    }

  }; // namespace Upmem
};