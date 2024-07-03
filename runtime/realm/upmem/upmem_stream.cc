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

#include "realm/upmem/upmem_stream.h"

namespace Realm {

extern Logger log_xd;
extern Logger log_taskreg;

namespace Upmem {
    extern Logger log_dpu;
    extern Logger log_stream;
    extern Logger log_dpudma;

    ////////////////////////////////////////////////////////////////////////
    //
    // class DPUStream

    DPUStream::DPUStream(DPU* _dpu, DPUWorker* _worker /*, int rel_priority = 0*/)
        : dpu(_dpu)
        , worker(_worker)
        , issuing_copies(false)
    {
        assert(worker != 0);

        log_stream.info() << "stream created: dpu=" << dpu << " stream=" << stream;
    }

    DPUStream::~DPUStream(void) { DPU_ASSERT(dpu_free(*stream)); }

    DPU* DPUStream::get_dpu(void) const { return dpu; }

    struct dpu_set_t* DPUStream::get_stream(void) const { return stream; }

    void DPUStream::set_stream(struct dpu_set_t* _stream) { stream = _stream; }

    // may be called by anybody to enqueue a copy or an event
    void DPUStream::add_copy(DPUMemcpy* copy)
    {
        assert(0 && "hit old copy path"); // shouldn't be used any more
        bool add_to_worker = false;
        {
            AutoLock<> al(mutex);

            // if we didn't already have work AND if there's not an active
            //  worker issuing copies, request attention
            add_to_worker = (pending_copies.empty() && pending_events.empty() && !issuing_copies);

            pending_copies.push_back(copy);
        }

        if (add_to_worker)
            worker->add_stream(this);
    }

    void DPUStream::add_fence(DPUWorkFence* fence)
    {
        upmemEvent_t e = dpu->event_pool.get_event();

        // DPU_ASSERT(upmemEventRecord(e, stream));

        log_stream.debug() << "UPMEM fence event " << e << " recorded on stream " << stream
                           << " (DPU " << dpu << ")";

        add_event(e, fence, 0, 0);
    }

    void DPUStream::add_start_event(DPUWorkStart* start)
    {
        upmemEvent_t e = dpu->event_pool.get_event();

        // DPU_ASSERT(upmemEventRecord(e, stream));

        log_stream.debug() << "UPMEM start event " << e << " recorded on stream " << stream
                           << " (DPU " << dpu << ")";

        // record this as a start event
        add_event(e, 0, 0, start);
    }

    void DPUStream::add_notification(DPUCompletionNotification* notification)
    {
        upmemEvent_t e = dpu->event_pool.get_event();

        // DPU_ASSERT(upmemEventRecord(e, stream));

        add_event(e, 0, notification, 0);
    }

    void DPUStream::add_event(upmemEvent_t event, DPUWorkFence* fence,
        DPUCompletionNotification* notification,
        DPUWorkStart* start)
    {
        bool add_to_worker = false;
        {
            AutoLock<> al(mutex);

            // if we didn't already have work AND if there's not an active
            //  worker issuing copies, request attention
            add_to_worker = (pending_copies.empty() && pending_events.empty() && !issuing_copies);

            PendingEvent e;
            e.event = event;
            e.fence = fence;
            e.start = start;
            e.notification = notification;

            pending_events.push_back(e);
        }

        if (add_to_worker)
            worker->add_stream(this);
    }

    void DPUStream::wait_on_streams(const std::set<DPUStream*>& other_streams)
    {
        assert(!other_streams.empty());
        for (std::set<DPUStream*>::const_iterator it = other_streams.begin();
             it != other_streams.end(); it++) {
            if (*it == this)
                continue;
            upmemEvent_t e = dpu->event_pool.get_event();

            log_stream.debug() << "UPMEM stream " << stream << " waiting on stream "
                               << (*it)->get_stream() << " (DPU " << dpu << ")";
            // sync
            DPU_ASSERT(dpu_sync(*(*it)->get_stream()));

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
    bool DPUStream::ok_to_submit_copy(size_t bytes, XferDes* xd) { return true; }

    // to be called by a worker (that should already have the DPU context
    //   current) - returns true if any work remains
    bool DPUStream::issue_copies(TimeLimit work_until)
    {
        // we have to make sure copies for a given stream are issued
        //  in order, so grab the thing at the front of the queue, but
        //  also set a flag taking ownersupmem of the head of the queue
        DPUMemcpy* copy = 0;
        {
            AutoLock<> al(mutex);

            // if the flag is set, we can't do any copies
            if (issuing_copies || pending_copies.empty()) {
                // no copies left, but stream might have other work left
                return has_work();
            }

            copy = pending_copies.front();
            pending_copies.pop_front();
            issuing_copies = true;
        }

        while (true) {
            {
                copy->execute(this);
            }

            // TODO: recycle these
            delete copy;

            // don't take another copy (but do clear the ownersupmem flag)
            //  if we're out of time
            bool expired = work_until.is_expired();

            {
                AutoLock<> al(mutex);

                if (pending_copies.empty()) {
                    issuing_copies = false;
                    // no copies left, but stream might have other work left
                    return has_work();
                } else {
                    if (expired) {
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

            if (pending_events.empty())
                // no events left, but stream might have other work left
                return has_work();

            event = pending_events.front().event;
            event_valid = true;
        }

        // we'll keep looking at events until we find one that hasn't triggered
        bool work_left = true;
        while (event_valid) {
            //   dpu_error_t res = upmemEventQuery(event);

            //   if(res == upmemErrorNotReady)
            //     return true; // oldest event hasn't triggered - check again later

            //   // no other kind of error is expected
            //   if(res != DPU_OK) {
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
            DPUWorkFence* fence = 0;
            DPUWorkStart* start = 0;
            DPUCompletionNotification* notification = 0;

            {
                AutoLock<> al(mutex);

                const PendingEvent& e = pending_events.front();
                assert(e.event == event);
                fence = e.fence;
                start = e.start;
                notification = e.notification;
                pending_events.pop_front();

                if (pending_events.empty()) {
                    event_valid = false;
                    work_left = has_work();
                } else
                    event = pending_events.front().event;
            }

            if (start) {
                start->mark_dpu_work_start();
            }
            if (fence)
                fence->mark_finished(true /*successful*/);

            if (notification)
                notification->request_completed();

            // don't repeat if we're out of time
            if (event_valid && work_until.is_expired())
                return true;
        }

        // if we get here, we ran out of events, but there might have been
        //  other kinds of work that we need to let the caller know about
        return work_left;
    }

}; // namespace Upmem
}; // namespace Realm