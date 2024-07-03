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

#ifndef REALM_UPMEM_EVENTS_H
#define REALM_UPMEM_EVENTS_H

// #include "realm/upmem/upmem_internal.h"

#include "realm/event_impl.h"

#include "realm/bgwork.h"
#include "realm/mem_impl.h"
#include "realm/proc_impl.h"
#include "realm/transfer/channel.h"
#include "realm/transfer/ib_memory.h"

namespace Realm {
namespace Upmem {
    // forward declarations
    // internal.h
    class DPU;

#ifndef EVENT_T
#define EVENT_T
    typedef std::string upmemEvent_t;
#endif

    void upmemEventCreate(upmemEvent_t* e);
    void upmemEventDestroy(upmemEvent_t* e);

    // a little helper class to manage a pool of CUevents that can be reused
    //  to reduce alloc/destroy overheads
    class DPUEventPool {
    public:
        DPUEventPool(int _batch_size = 256);

        // allocating the initial batch of events and cleaning up are done with
        //  these methods instead of constructor/destructor because we don't
        //  manage the DPU context in this helper class
        void init_pool(int init_size = 0 /* default == batch size */);
        void empty_pool(void);

        upmemEvent_t get_event(bool external = false);
        void return_event(upmemEvent_t e, bool external = false);

    protected:
        Mutex mutex;
        int batch_size, current_size, total_size, external_count;
        std::vector<upmemEvent_t> available_events;

    }; // end class DPUEventPool

    // an interface for receiving completion notification for a DPU operation
    //  (right now, just copies)
    class DPUCompletionNotification {
    public:
        virtual ~DPUCompletionNotification(void) { }

        virtual void request_completed(void) = 0;
    }; // end class DPUCompletionNotification

    class DPUPreemptionWaiter : public DPUCompletionNotification {
    public:
        DPUPreemptionWaiter(DPU* dpu);
        virtual ~DPUPreemptionWaiter(void) { }

    public:
        virtual void request_completed(void);

    public:
        void preempt(void);

    private:
        DPU* const dpu;
        Event wait_event;
    }; // end class DPUPreemptionWaiter

    class DPURequest;

    class DPUCompletionEvent : public DPUCompletionNotification {
    public:
        void request_completed(void);

        DPURequest* req;
    }; // end class DPUCompletionEvent

    class DPURequest : public Request {
    public:
        const void* src_base;
        void* dst_base;
        DPU* dst_dpu;
        DPUCompletionEvent event;
    }; // end class DPURequest

}; // namespace Upmem
}; // namespace Realm

#endif
