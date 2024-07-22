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

#include "realm/upmem/upmem_workers.h"
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
    // class DPUWorker

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
    // class BlockingCompletionNotification

    class BlockingCompletionNotification : public DPUCompletionNotification {
    public:
      BlockingCompletionNotification(void);
      virtual ~BlockingCompletionNotification(void);

      virtual void request_completed(void);

      virtual void wait(void);

    public:
      atomic<bool> completed;
    };

    BlockingCompletionNotification::BlockingCompletionNotification(void)
      : completed(false)
    {}

    BlockingCompletionNotification::~BlockingCompletionNotification(void) {}

    void BlockingCompletionNotification::request_completed(void)
    {
      // no condition variable needed - the waiter is spinning
      completed.store(true);
    }

    void BlockingCompletionNotification::wait(void)
    {
      // blocking completion is horrible and should die as soon as possible
      // in the mean time, we need to assist with background work to avoid
      //  the risk of deadlock
      // note that this means you can get NESTED blocking completion
      //  notifications, which is just one of the ways this is horrible
      BackgroundWorkManager::Worker worker;

      worker.set_manager(&(get_runtime()->bgwork));

      while(!completed.load())
        worker.do_work(-1 /* as long as it takes */, &completed /* until this is set */);
    }

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
        CHECK_UPMEM(dpu_callback(
            *stream->get_stream(), &upmem_start_callback, (void *)this,
            (dpu_callback_flags_t)(DPU_CALLBACK_ASYNC | DPU_CALLBACK_NONBLOCKING)));
      } else {
        assert(0 && "cfg_fences_use_callbacks must be set true");
        stream->add_fence(this);
      }
    }

    // callback this function when work is finished
    // mark finished on callback
    /*static*/ dpu_error_t DPUWorkFence::upmem_start_callback(struct dpu_set_t stream,
                                                              uint32_t rank_id,
                                                              void *data)
    {
      DPUWorkFence *me = (DPUWorkFence *)data;

      me->mark_finished(true /*succesful*/);
      return DPU_OK;
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
        CHECK_UPMEM(dpu_callback(
            *stream->get_stream(), &upmem_start_callback, (void *)this,
            (dpu_callback_flags_t)(DPU_CALLBACK_ASYNC | DPU_CALLBACK_NONBLOCKING)));
      } else {
        assert(0 && "cfg_fences_use_callbacks must be set true");
        stream->add_start_event(this);
      }
    }

    void DPUWorkStart::mark_dpu_work_start()
    {
      // op is operater from Realm::Operation
      op->mark_gpu_work_start();
      mark_finished(true);
    }

    /*static*/ dpu_error_t DPUWorkStart::upmem_start_callback(struct dpu_set_t stream,
                                                              uint32_t rank_id,
                                                              void *data)
    {
      DPUWorkStart *me = (DPUWorkStart *)data;
      // record the real start time for the operation
      me->mark_dpu_work_start();
      return DPU_OK;
    }

  }; // namespace Upmem
};   // namespace Realm