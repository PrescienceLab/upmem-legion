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
#include "realm/upmem/upmem_module.h"

namespace Realm {

  extern Logger log_xd;
  extern Logger log_taskreg;

  namespace Upmem {
    extern Logger log_dpu;
    extern Logger log_stream;
    extern Logger log_dpudma;

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
      push_context(); // currently doesn't do anything. we have one device atm
      event_pool.init_pool();

      // DPUstreams are just DPU_sets

      task_streams.resize(module->config->cfg_task_streams);
      for(unsigned i = 0; i < module->config->cfg_task_streams; i++)
        task_streams[i] = new DPUStream(this, worker);

      pop_context();
    }

    DPU::~DPU(void)
    {
      push_context();
      event_pool.empty_pool();
    }

    void DPU::push_context(void)
    {
      // CHECK_HIP( hipSetDevice(device_id) );
    }

    void DPU::pop_context(void)
    {
      // the context we pop had better be ours...
      // hipCtx_t popped;
      // CHECK_HIP( hipCtxPopCurrent(&popped) );
      // assert(popped == context);
    }

    ///////////////////////////////////////////////////////////////////////
    //
    // class DPUProcessor

    DPUProcessor::DPUProcessor(DPU *_dpu, Processor _me, Realm::CoreReservationSet &crs,
                               size_t _stack_size)
      : LocalTaskProcessor(_me, Processor::TOC_PROC)
      , dpu(_dpu)
      , block_on_synchronize(false)
      , ctxsync(_dpu, _dpu->device_id, crs, _dpu->module->config->cfg_max_ctxsync_threads)
    {
      Realm::CoreReservationParameters params;
      params.set_num_cores(1);
      params.set_alu_usage(params.CORE_USAGE_SHARED);
      params.set_fpu_usage(params.CORE_USAGE_SHARED);
      params.set_ldst_usage(params.CORE_USAGE_SHARED);
      params.set_max_stack_size(_stack_size);

      std::string name = stringbuilder() << "DPU proc " << _me;

      core_rsrv = new Realm::CoreReservation(name, crs, params);

#ifdef REALM_USE_USER_THREADS_FOR_DPU
      Realm::UserThreadTaskScheduler *sched =
          new DPUTaskScheduler<Realm::UserThreadTaskScheduler>(me, *core_rsrv, this);
#else
      Realm::KernelThreadTaskScheduler *sched =
          new DPUTaskScheduler<Realm::KernelThreadTaskScheduler>(me, *core_rsrv, this);
#endif
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

      // // if we don't have a function pointer implementation, see if we can make one
      // if(!fpi) {
      //   const std::vector<CodeTranslator *>& translators =
      //   get_runtime()->get_code_translators(); for(std::vector<CodeTranslator
      //   *>::const_iterator it = translators.begin();
      //       it != translators.end();
      //       it++)
      //     if((*it)->can_translate<FunctionPointerImplementation>(codedesc)) {
      //       FunctionPointerImplementation *newfpi =
      //       (*it)->translate<FunctionPointerImplementation>(codedesc); if(newfpi) {
      //         log_taskreg.info() << "function pointer created: trans=" << (*it)->name
      //         << " fnptr=" << (void *)(newfpi->fnptr);
      //         codedesc.add_implementation(newfpi);
      //         fpi = newfpi;
      //         break;
      //       }
      //     }
      // }

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
        dpu_set_t *stream = ThreadLocal::current_dpu_stream->get_stream();

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

      // shut down threads/scheduler
      LocalTaskProcessor::shutdown();

      ctxsync.shutdown_threads();

      // synchronize the device so we can flush any printf buffers - do
      //  this after shutting down the threads so that we know all work is done
      // {
      // AutoDPUContext agc(dpu);

      // CHECK_HIP( hipDeviceSynchronize() );
      // }
    }

    DPUWorker::DPUWorker(void)
      : BackgroundWorkItem("dpu worker")
      , condvar(lock)
      , core_rsrv(0)
      , worker_thread(0)
      , thread_sleeping(false)
      , worker_shutdown_requested(false)
    {}

    DPUWorker::~DPUWorker(void)
    {
      // shutdown should have already been called
      assert(worker_thread == 0);
    }

    void DPUWorker::start_background_thread(Realm::CoreReservationSet &crs,
                                            size_t stack_size)
    {
      // shouldn't be doing this if we've registered as a background work item
      assert(manager == 0);

      core_rsrv = new Realm::CoreReservation("DPU worker thread", crs,
                                             Realm::CoreReservationParameters());

      Realm::ThreadLaunchParameters tlp;

      worker_thread =
          Realm::Thread::create_kernel_thread<DPUWorker, &DPUWorker::thread_main>(
              this, tlp, *core_rsrv, 0);
    }

    void DPUWorker::shutdown_background_thread(void)
    {
      {
        AutoLock<> al(lock);
        worker_shutdown_requested.store(true);
        if(thread_sleeping) {
          thread_sleeping = false;
          condvar.broadcast();
        }
      }

      worker_thread->join();
      delete worker_thread;
      worker_thread = 0;

      delete core_rsrv;
      core_rsrv = 0;
    }

    void DPUWorker::add_stream(DPUStream *stream)
    {
      bool was_empty = false;
      {
        AutoLock<> al(lock);

#ifdef DEBUG_REALM
        // insist that the caller de-duplicate these
        for(ActiveStreamQueue::iterator it = active_streams.begin();
            it != active_streams.end(); ++it)
          assert(*it != stream);
#endif
        was_empty = active_streams.empty();
        active_streams.push_back(stream);

        if(thread_sleeping) {
          thread_sleeping = false;
          condvar.broadcast();
        }
      }

      // if we're a background work item, request attention if needed
      if(was_empty && (manager != 0))
        make_active();
    }

    bool DPUWorker::do_work(TimeLimit work_until)
    {
      // pop the first stream off the list and immediately become re-active
      //  if more streams remain
      DPUStream *stream = 0;
      bool still_not_empty = false;
      {
        AutoLock<> al(lock);

        assert(!active_streams.empty());
        stream = active_streams.front();
        active_streams.pop_front();
        still_not_empty = !active_streams.empty();
      }
      if(still_not_empty)
        make_active();

      // do work for the stream we popped, paying attention to the cutoff
      //  time
      bool requeue_stream = false;

      if(stream->reap_events(work_until)) {
        // still work (e.g. copies) to do
        if(work_until.is_expired()) {
          // out of time - save it for later
          requeue_stream = true;
        } else {
          if(stream->issue_copies(work_until))
            requeue_stream = true;
        }
      }

      bool was_empty = false;
      if(requeue_stream) {
        AutoLock<> al(lock);

        was_empty = active_streams.empty();
        active_streams.push_back(stream);
      }
      // note that we can need requeueing even if we called make_active above!
      return was_empty;
    }

    bool DPUWorker::process_streams(bool sleep_on_empty)
    {
      DPUStream *cur_stream = 0;
      DPUStream *first_stream = 0;
      bool requeue_stream = false;

      while(true) {
        // grab the front stream in the list
        {
          AutoLock<> al(lock);

          // if we didn't finish work on the stream from the previous
          //  iteration, add it back to the end
          if(requeue_stream)
            active_streams.push_back(cur_stream);

          while(active_streams.empty()) {
            // sleep only if this was the first attempt to get a stream
            if(sleep_on_empty && (first_stream == 0) &&
               !worker_shutdown_requested.load()) {
              thread_sleeping = true;
              condvar.wait();
            } else
              return false;
          }

          cur_stream = active_streams.front();
          // did we wrap around?  if so, stop for now
          if(cur_stream == first_stream)
            return true;

          active_streams.pop_front();
          if(!first_stream)
            first_stream = cur_stream;
        }

        // and do some work for it
        requeue_stream = false;

        // both reap_events and issue_copies report whether any kind of work
        //  remains, so we have to be careful to avoid double-requeueing -
        //  if the first call returns false, we can't try the second one
        //  because we may be doing (or failing to do and then requeuing)
        //  somebody else's work
        if(!cur_stream->reap_events(TimeLimit()))
          continue;
        if(!cur_stream->issue_copies(TimeLimit()))
          continue;

        // if we fall all the way through, the queues never went empty at
        //  any time, so it's up to us to requeue
        requeue_stream = true;
      }
    }

    void DPUWorker::thread_main(void)
    {
      // TODO: consider busy-waiting in some cases to reduce latency?
      while(!worker_shutdown_requested.load()) {
        bool work_left = process_streams(true);

        // if there was work left, yield our thread for now to avoid a tight spin loop
        // TODO: enqueue a callback so we can go to sleep and wake up sooner than a kernel
        //  timeslice?
        if(work_left)
          Realm::Thread::yield();
      }
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUStream

    DPUStream::DPUStream(DPU *_dpu, DPUWorker *_worker /*, int rel_priority = 0*/)
      : dpu(_dpu)
      , worker(_worker)
      , issuing_copies(false)
    {
      assert(worker != 0);
      // AT THE MOMENT, STREAMS ARE JUST DPU SETS WITH 64 DPUS
      DPU_ASSERT(dpu_alloc(64, NULL, stream));
      log_stream.info() << "stream created: dpu=" << dpu << " stream=" << stream;
    }

    DPUStream::~DPUStream(void) { DPU_ASSERT(dpu_free(*stream)); }

    DPU *DPUStream::get_dpu(void) const { return dpu; }

    dpu_set_t *DPUStream::get_stream(void) const { return stream; }

    // may be called by anybody to enqueue a copy or an event
    void DPUStream::add_copy(DPUMemcpy *copy)
    {
      assert(0 && "hit old copy path"); // shouldn't be used any more
      bool add_to_worker = false;
      {
        AutoLock<> al(mutex);

        // if we didn't already have work AND if there's not an active
        //  worker issuing copies, request attention
        add_to_worker =
            (pending_copies.empty() && pending_events.empty() && !issuing_copies);

        pending_copies.push_back(copy);
      }

      if(add_to_worker)
        worker->add_stream(this);
    }

    void DPUStream::add_fence(DPUWorkFence *fence)
    {
      upmemEvent_t e = dpu->event_pool.get_event();

      DPU_ASSERT(upmemEventRecord(e, stream));

      log_stream.debug() << "UPMEM fence event " << e << " recorded on stream " << stream
                         << " (DPU " << dpu << ")";

      add_event(e, fence, 0, 0);
    }

    void DPUStream::add_start_event(DPUWorkStart *start)
    {
      upmemEvent_t e = dpu->event_pool.get_event();

      DPU_ASSERT(upmemEventRecord(e, stream));

      log_stream.debug() << "UPMEM start event " << e << " recorded on stream " << stream
                         << " (DPU " << dpu << ")";

      // record this as a start event
      add_event(e, 0, 0, start);
    }

    void DPUStream::add_notification(DPUCompletionNotification *notification)
    {
      upmemEvent_t e = dpu->event_pool.get_event();

      DPU_ASSERT(upmemEventRecord(e, stream));

      add_event(e, 0, notification, 0);
    }

    void DPUStream::add_event(upmemEvent_t event, DPUWorkFence *fence,
                              DPUCompletionNotification *notification,
                              DPUWorkStart *start)
    {
      bool add_to_worker = false;
      {
        AutoLock<> al(mutex);

        // if we didn't already have work AND if there's not an active
        //  worker issuing copies, request attention
        add_to_worker =
            (pending_copies.empty() && pending_events.empty() && !issuing_copies);

        PendingEvent e;
        e.event = event;
        e.fence = fence;
        e.start = start;
        e.notification = notification;

        pending_events.push_back(e);
      }

      if(add_to_worker)
        worker->add_stream(this);
    }

    void DPUStream::wait_on_streams(const std::set<DPUStream *> &other_streams)
    {
      assert(!other_streams.empty());
      for(std::set<DPUStream *>::const_iterator it = other_streams.begin();
          it != other_streams.end(); it++) {
        if(*it == this)
          continue;
        upmemEvent_t e = dpu->event_pool.get_event();

        DPU_ASSERT(upmemEventRecord(e, (*it)->get_stream()));

        log_stream.debug() << "UPMEM stream " << stream << " waiting on stream "
                           << (*it)->get_stream() << " (DPU " << dpu << ")";

        DPU_ASSERT(upmemStreamWaitEvent(stream, e, 0));

        // record this event on our stream
        add_event(e, 0);
      }
    }

    bool DPUStream::has_work(void) const
    {
      return (!pending_events.empty() || !pending_copies.empty());
    }

    // atomically checks rate limit counters and returns true if 'bytes'
    //  worth of copies can be submitted or false if not (in which case
    //  the progress counter on the xd will be updated when it should try
    //  again)
    bool DPUStream::ok_to_submit_copy(size_t bytes, XferDes *xd) { return true; }

    // to be called by a worker (that should already have the DPU context
    //   current) - returns true if any work remains
    bool DPUStream::issue_copies(TimeLimit work_until)
    {
      // we have to make sure copies for a given stream are issued
      //  in order, so grab the thing at the front of the queue, but
      //  also set a flag taking ownersupmem of the head of the queue
      DPUMemcpy *copy = 0;
      {
        AutoLock<> al(mutex);

        // if the flag is set, we can't do any copies
        if(issuing_copies || pending_copies.empty()) {
          // no copies left, but stream might have other work left
          return has_work();
        }

        copy = pending_copies.front();
        pending_copies.pop_front();
        issuing_copies = true;
      }

      while(true) {
        {
          AutoDPUContext agc(dpu);
          copy->execute(this);
        }

        // TODO: recycle these
        delete copy;

        // don't take another copy (but do clear the ownersupmem flag)
        //  if we're out of time
        bool expired = work_until.is_expired();

        {
          AutoLock<> al(mutex);

          if(pending_copies.empty()) {
            issuing_copies = false;
            // no copies left, but stream might have other work left
            return has_work();
          } else {
            if(expired) {
              issuing_copies = false;
              // definitely still work to do
              return true;
            } else {
              // take the next copy
              copy = pending_copies.front();
              pending_copies.pop_front();
            }
          }
        }
      }
    }

    bool DPUStream::reap_events(TimeLimit work_until)
    {
      // peek at the first event
      upmemEvent_t event;
      bool event_valid = false;
      {
        AutoLock<> al(mutex);

        if(pending_events.empty())
          // no events left, but stream might have other work left
          return has_work();

        event = pending_events.front().event;
        event_valid = true;
      }

      // we'll keep looking at events until we find one that hasn't triggered
      bool work_left = true;
      while(event_valid) {
        dpu_error_t res = upmemEventQuery(event);

        //   if(res == upmemErrorNotReady)
        //     return true; // oldest event hasn't triggered - check again later

        //   // no other kind of error is expected
        //   if(res != upmemSuccess) {
        //     const char *ename = 0;
        //     const char *estr = 0;
        //     ename = upmemGetErrorName(res);
        //     estr = upmemGetErrorString(res);
        //     log_dpu.fatal() << "UPMEM error reported on DPU " << dpu->info->index << ":
        //     "
        //                     << estr << " (" << ename << ")";
        //     assert(0);
        //   }

        log_stream.debug() << "UPMEM event " << event << " triggered on stream " << stream
                           << " (DPU " << dpu << ")";

        // give event back to DPU for reuse
        dpu->event_pool.return_event(event);

        // this event has triggered, so figure out the fence/notification to trigger
        //  and also peek at the next event
        DPUWorkFence *fence = 0;
        DPUWorkStart *start = 0;
        DPUCompletionNotification *notification = 0;

        {
          AutoLock<> al(mutex);

          const PendingEvent &e = pending_events.front();
          assert(e.event == event);
          fence = e.fence;
          start = e.start;
          notification = e.notification;
          pending_events.pop_front();

          if(pending_events.empty()) {
            event_valid = false;
            work_left = has_work();
          } else
            event = pending_events.front().event;
        }

        if(start) {
          start->mark_dpu_work_start();
        }
        if(fence)
          fence->mark_finished(true /*successful*/);

        if(notification)
          notification->request_completed();

        // don't repeat if we're out of time
        if(event_valid && work_until.is_expired())
          return true;
      }

      // if we get here, we ran out of events, but there might have been
      //  other kinds of work that we need to let the caller know about
      return work_left;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUEventPool

    DPUEventPool::DPUEventPool(int _batch_size)
      : batch_size(_batch_size)
      , current_size(0)
      , total_size(0)
      , external_count(0)
    {
      // don't immediately fill the pool because we're not managing the context ourselves
    }

    // allocating the initial batch of events and cleaning up are done with
    //  these methods instead of constructor/destructor because we don't
    //  manage the DPU context in this helper class
    void DPUEventPool::init_pool(int init_size /*= 0 -- default == batch size */)
    {
      assert(available_events.empty());

      if(init_size == 0)
        init_size = batch_size;

      available_events.resize(init_size);

      current_size = init_size;
      total_size = init_size;

      // TODO: measure how much benefit is derived from CU_EVENT_DISABLE_TIMING and
      //  consider using them for completion callbacks
      for(int i = 0; i < init_size; i++)
        CHECK_HIP(hipEventCreateWithFlags(&available_events[i], hipEventDefault));
    }

    void DPUEventPool::empty_pool(void)
    {
      // shouldn't be any events running around still
      assert((current_size + external_count) == total_size);
      if(external_count)
        log_stream.warning() << "Application leaking " << external_count
                             << " cuda events";

      for(int i = 0; i < current_size; i++)
        CHECK_HIP(hipEventDestroy(available_events[i]));

      current_size = 0;
      total_size = 0;

      // free internal vector storage
      std::vector<upmemEvent_t>().swap(available_events);
    }

    upmemEvent_t DPUEventPool::get_event(bool external)
    {
      AutoLock<> al(mutex);

      if(current_size == 0) {
        // if we need to make an event, make a bunch
        current_size = batch_size;
        total_size += batch_size;

        log_stream.info() << "event pool " << this << " depleted - adding " << batch_size
                          << " events";

        // resize the vector (considering all events that might come back)
        available_events.resize(total_size);

        for(int i = 0; i < batch_size; i++)
          CHECK_HIP(hipEventCreateWithFlags(&available_events[i], hipEventDefault));
      }

      if(external)
        external_count++;

      return available_events[--current_size];
    }

    void DPUEventPool::return_event(upmemEvent_t e, bool external)
    {
      AutoLock<> al(mutex);

      assert(current_size < total_size);

      if(external) {
        assert(external_count);
        external_count--;
      }

      available_events[current_size++] = e;
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUMemcpy

    DPUMemcpy::DPUMemcpy(DPU *_dpu, DPUMemcpyKind _kind)
      : dpu(_dpu)
      , kind(_kind)
    {}

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUWorkFence

    DPUWorkFence::DPUWorkFence(Realm::Operation *op)
      : Realm::Operation::AsyncWorkItem(op)
    {}

    void DPUWorkFence::request_cancellation(void)
    {
      // ignored - no way to shoot down UPMEM work
    }

    void DPUWorkFence::print(std::ostream &os) const { os << "DPUWorkFence"; }

    void DPUWorkFence::enqueue_on_stream(DPUStream *stream)
    {

      if(stream->get_dpu()->module->config->cfg_fences_use_callbacks) {
        DPU_ASSERT(dpu_callback(stream->get_stream(), &upmem_start_callback, (void *)this,
                                DPU_CALLBACK_NONBLOCKING));
      } else {
        stream->add_fence(this);
      }
    }

    // callback this function when work is finished
    // mark finished on callback
    /*static*/ void DPUWorkFence::dpu_callback(dpu_set_t stream, uint32_t rank_id,
                                               void *data)
    {
      DPUWorkFence *me = (DPUWorkFence *)data;

      // assert(res == upmemSuccess);
      me->mark_finished(true /*succesful*/);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUWorkStart
    DPUWorkStart::DPUWorkStart(Realm::Operation *op)
      : Realm::Operation::AsyncWorkItem(op)
    {}

    void DPUWorkStart::print(std::ostream &os) const { os << "DPUWorkStart"; }

    void DPUWorkStart::enqueue_on_stream(DPUStream *stream)
    {
      if(stream->get_dpu()->module->config->cfg_fences_use_callbacks) {
        DPU_ASSERT(dpu_callback(*stream->get_stream(), &upmem_start_callback,
                                (void *)this, DPU_CALLBACK_NONBLOCKING));
      } else {
        stream->add_start_event(this);
      }
    }

    void DPUWorkStart::mark_dpu_work_start()
    {
      // op is operater from Realm::Operation
      op->mark_gpu_work_start();
      mark_finished(true);
    }

    /*static*/ void DPUWorkStart::upmem_start_callback(dpu_set_t stream, uint32_t rank_id,
                                                       void *data)
    {
      DPUWorkStart *me = (DPUWorkStart *)data;
      // assert(res == upmemSuccess);
      // record the real start time for the operation
      me->mark_dpu_work_start();
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUMemcpyFence

    DPUMemcpyFence::DPUMemcpyFence(DPU *_dpu, DPUMemcpyKind _kind, DPUWorkFence *_fence)
      : DPUMemcpy(_dpu, _kind)
      , fence(_fence)
    {
      // log_stream.info() << "dpu memcpy fence " << this << " (fence = " << fence << ")
      // created";
    }

    void DPUMemcpyFence::execute(DPUStream *stream)
    {
      // log_stream.info() << "dpu memcpy fence " << this << " (fence = " << fence << ")
      // executed";
      fence->enqueue_on_stream(stream);
#ifdef FORCE_DPU_STREAM_SYNCHRONIZE
      DPU_ASSERT(dpu_sync(*stream->get_stream()));
#endif
    }

    DPUPreemptionWaiter::DPUPreemptionWaiter(DPU *g)
      : dpu(g)
    {
      GenEventImpl *impl = GenEventImpl::create_genevent();
      wait_event = impl->current_event();
    }

    void DPUPreemptionWaiter::request_completed(void)
    {
      GenEventImpl::trigger(wait_event, false /*poisoned*/);
    }

    void DPUPreemptionWaiter::preempt(void)
    {
      // Realm threads don't obey a stack discipline for
      // preemption so we can't leave our context on the stack
      dpu->pop_context();
      wait_event.wait();
      // When we wake back up, we have to push our context again
      dpu->push_context();
    }
    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUCompletionEvent

    void DPUCompletionEvent::request_completed(void)
    {
      req->xd->notify_request_read_done(req);
      req->xd->notify_request_write_done(req);
    }

    ////////////////////////////////////////////////////////////////////////
    //
    // class AutoDPUContext

    AutoDPUContext::AutoDPUContext(DPU &_dpu)
      : dpu(&_dpu)
    {
      dpu->push_context();
    }

    AutoDPUContext::AutoDPUContext(DPU *_dpu)
      : dpu(_dpu)
    {
      if(dpu)
        dpu->push_context();
    }

    AutoDPUContext::~AutoDPUContext(void)
    {
      if(dpu)
        dpu->pop_context();
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

        //   dpu_error_t ret;
        //   {
        //     AutoDPUContext agc(module->dpus[0]);
        //     ret = upmemHostRegister(base, bytes,
        //                           upmemHostRegisterPortable |
        //                           upmemHostRegisterMapped);
        //   }
        //   if(ret != upmemSuccess) {
        //     log_dpu.fatal() << "failed to register replicated heap chunk: base=" <<
        //     base
        //                     << " size=" << bytes << " ret=" << ret;
        //     abort();
      }
    }

    void DPUReplHeapListener::chunk_destroyed(void *base, size_t bytes)
    {
      if(!module->dpus.empty()) {
        log_dpu.info() << "unregistering replicated heap chunk: base=" << base
                       << " size=" << bytes;

        //   dpu_error_t ret;
        //   {
        //     AutoDPUContext agc(module->dpus[0]);
        //     ret = upmemHostUnregister(base);
        //   }
        //   if(ret != upmemSuccess)
        //     log_dpu.warning() << "failed to unregister replicated heap chunk: base=" <<
        //     base
        //                       << " size=" << bytes << " ret=" << ret;
      }
    }

  }; // namespace Upmem

}; // namespace Realm