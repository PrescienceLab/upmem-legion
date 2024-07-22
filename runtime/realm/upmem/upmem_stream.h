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

#ifndef REALM_UPMEM_STREAM_H
#define REALM_UPMEM_STREAM_H

#include "realm/circ_queue.h"

#include "realm/bgwork.h"
#include "realm/mem_impl.h"
#include "realm/proc_impl.h"
#include "realm/transfer/channel.h"
#include "realm/transfer/ib_memory.h"

#include "realm/upmem/upmem_events.h"
#include "realm/upmem/upmem_memory.h"
#include "realm/upmem/upmem_workers.h"

namespace Realm {
  namespace Upmem {
    // forward declarations
    // internal.h
    class DPU;
    // memory.h
    class DPUMemcpy;
    // workers.h
    class DPUWorker;
    class DPUWorkFence;
    class DPUWorkStart;
    // events.h
    class DPUCompletionNotification;

#ifndef EVENT_T
#define EVENT_T
    typedef std::string upmemEvent_t;
#endif

    class DPUStream {
    public:
      DPUStream(DPU *_dpu, DPUWorker *_worker /*,  int rel_priority = 0 */);
      ~DPUStream(void);

      DPU *get_dpu(void) const;
      struct dpu_set_t *get_stream(void) const;
      void set_stream(struct dpu_set_t *stream);

      // may be called by anybody to enqueue a copy or an event
      void add_copy(DPUMemcpy *copy);
      void add_fence(DPUWorkFence *fence);
      void add_start_event(DPUWorkStart *start);
      void add_notification(DPUCompletionNotification *notification);
      void wait_on_streams(const std::set<DPUStream *> &other_streams);

      // atomically checks rate limit counters and returns true if 'bytes'
      //  worth of copies can be submitted or false if not (in which case
      //  the progress counter on the xd will be updated when it should try
      //  again)
      bool ok_to_submit_copy(size_t bytes, XferDes *xd);

      // to be called by a worker (that should already have the DPU context
      //   current) - returns true if any work remains
      bool issue_copies(TimeLimit work_until);
      bool reap_events(TimeLimit work_until);

    protected:
      // may only be tested with lock held
      bool has_work(void) const;

      void add_event(upmemEvent_t event, DPUWorkFence *fence,
                     DPUCompletionNotification *notification = NULL,
                     DPUWorkStart *start = NULL);

      DPU *dpu;
      DPUWorker *worker;

      struct dpu_set_t *stream;

      Mutex mutex;

#define USE_CQ
#ifdef USE_CQ
      Realm::CircularQueue<DPUMemcpy *> pending_copies;
#else
      std::deque<DPUMemcpy *> pending_copies;
#endif
      bool issuing_copies;

      struct PendingEvent {
        upmemEvent_t event;
        DPUWorkFence *fence;
        DPUWorkStart *start;
        DPUCompletionNotification *notification;
      };
#ifdef USE_CQ
      Realm::CircularQueue<PendingEvent> pending_events;
#else
      std::deque<PendingEvent> pending_events;
#endif

    }; // end class DPUStream

  }; // namespace Upmem
};   // namespace Realm

#endif
