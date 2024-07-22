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

#ifndef REALM_UPMEM_WORKERS_H
#define REALM_UPMEM_WORKERS_H

#ifndef DPURT
#define DPURT
#include <dpu> // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

#include "realm/circ_queue.h"

#include "realm/upmem/upmem_stream.h"

namespace Realm {
  namespace Upmem {
    // forward declaration
    // stream.h
    class DPUStream;

    // a DPUWorker is responsible for making progress on one or more DPUStreams -
    //  this may be done directly by a DPUProcessor or in a background thread
    //  spawned for the purpose
    class DPUWorker : public BackgroundWorkItem {
    public:
      DPUWorker(void);
      virtual ~DPUWorker(void);

      // adds a stream that has work to be done
      void add_stream(DPUStream *s);

      // used to start a dedicate thread (mutually exclusive with being
      //  registered with a background work manager)
      void start_background_thread(Realm::CoreReservationSet &crs, size_t stack_size);
      void shutdown_background_thread(void);

      bool do_work(TimeLimit work_until);

    public:
      void thread_main(void);

    protected:
      // used by the background thread
      // processes work on streams, optionally sleeping for work to show up
      // returns true if work remains to be done
      bool process_streams(bool sleep_on_empty);

      Mutex lock;
      Mutex::CondVar condvar;

      typedef CircularQueue<DPUStream *, 16> ActiveStreamQueue;
      ActiveStreamQueue active_streams;

      // used by the background thread (if any)
      Realm::CoreReservation *core_rsrv;
      Realm::Thread *worker_thread;
      bool thread_sleeping;
      atomic<bool> worker_shutdown_requested;

    }; // end class DPUWorker

    class DPUWorkFence : public Realm::Operation::AsyncWorkItem {
    public:
      DPUWorkFence(Realm::Operation *op);

      virtual void request_cancellation(void);

      void enqueue_on_stream(DPUStream *stream);

      virtual void print(std::ostream &os) const;

      IntrusiveListLink<DPUWorkFence> fence_list_link;
      REALM_PMTA_DEFN(DPUWorkFence, IntrusiveListLink<DPUWorkFence>, fence_list_link);
      typedef IntrusiveList<DPUWorkFence, REALM_PMTA_USE(DPUWorkFence, fence_list_link),
                            DummyLock>
          FenceList;

    protected:
      static dpu_error_t upmem_start_callback(struct dpu_set_t stream, uint32_t rank_id,
                                              void *data);
    }; // end class DPUWorkFence

    class DPUWorkStart : public Realm::Operation::AsyncWorkItem {
    public:
      DPUWorkStart(Realm::Operation *op);

      virtual void request_cancellation(void) { return; };

      void enqueue_on_stream(DPUStream *stream);

      virtual void print(std::ostream &os) const;

      void mark_dpu_work_start();

    protected:
      static dpu_error_t upmem_start_callback(struct dpu_set_t stream, uint32_t rank_id,
                                              void *data);
    }; // end class DPUWorkStart

  }; // namespace Upmem
};   // namespace Realm

#endif
