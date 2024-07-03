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

#include "realm/upmem/upmem_events.h"

namespace Realm {

  extern Logger log_xd;
  extern Logger log_taskreg;

  namespace Upmem {
    extern Logger log_dpu;
    extern Logger log_stream;
    extern Logger log_dpudma;

    void upmemEventCreate(upmemEvent_t *e) {}

    void upmemEventDestroy(upmemEvent_t *e) {}

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
    void DPUEventPool::init_pool(int init_size /* = 0 -- default == batch size */)
    {
      assert(available_events.empty());

      if(init_size == 0)
        init_size = batch_size;

      available_events.resize(init_size);

      current_size = init_size;
      total_size = init_size;

      for(int i = 0; i < init_size; i++) {
        // create events
        upmemEventCreate(&available_events[i]);
      }
    }

    void DPUEventPool::empty_pool(void)
    {
      // shouldn't be any events running around still
      assert((current_size + external_count) == total_size);
      if(external_count)
        log_stream.warning() << "Application leaking " << external_count
                             << " cuda events";

      for(int i = 0; i < current_size; i++) {
        // destroy all events
        upmemEventDestroy(&available_events[i]);
      }
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

        for(int i = 0; i < batch_size; i++) {
          upmemEventCreate(&available_events[i]);
        }
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
      wait_event.wait();
    }
    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUCompletionEvent

    void DPUCompletionEvent::request_completed(void)
    {
      req->xd->notify_request_read_done(req);
      req->xd->notify_request_write_done(req);
    }

  }; // namespace Upmem
};   // namespace Realm