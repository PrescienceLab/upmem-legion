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

#include "dpu.h" // UPMEM rt syslib

#include "realm/operation.h"
#include "realm/threads.h"
#include "realm/circ_queue.h"
#include "realm/indexspace.h"
#include "realm/proc_impl.h"
#include "realm/mem_impl.h"
#include "realm/bgwork.h"
#include "realm/transfer/channel.h"
#include "realm/transfer/ib_memory.h"

#define upememSuccess 1

namespace Realm {

  namespace Upmem {

    enum DPUMemcpyKind
    {
      DPU_MEMCPY_HOST_TO_DEVICE,
      DPU_MEMCPY_DEVICE_TO_HOST,
      DPU_MEMCPY_DEVICE_TO_DEVICE,
    };

    // Forard declaration
    class UpmemModule;
    class DPU;
    class DPUProcessor;
    class DPUWorker;
    class DPUStream;
    class DPURequest;
    class DPUCompletionEvent;
    class DPUCompletionNotification;

    // class DPUFBMemory;
    // class DPUZCMemory;
    // class DPUFBIBMemory;
    class UpmemModule;

    extern UpmemModule *upmem_module_singleton;

    class DPU {
    public:
      DPU(UpmemModule *_module, DPUInfo *_info, DPUWorker *worker, int _device_id);
      ~DPU(void);

      void push_context(void);
      void pop_context(void);

      void create_processor(RuntimeImpl *runtime, size_t stack_size);
      // void create_fb_memory(RuntimeImpl *runtime, size_t size, size_t ib_size);
      // void create_dynamic_fb_memory(RuntimeImpl *runtime, size_t max_size);
      // void create_dma_channels(Realm::RuntimeImpl *r);

    protected:
      upmemModule_t load_upmem_module(const void *data);

    public:
      UpmemModule *module;
      DPUInfo *info;
      DPUWorker *worker;
      DPUProcessor *proc;
      // DPUFBMemory *fbmem;
      // DPUFBIBMemory *fb_ibmem;

      // upmemCtx_t context;
      int device_id;
      char *fbmem_base, *fb_ibmem_base;

      // which system memories have been registered and can be used for cuMemcpyAsync
      // std::set<Memory> pinned_sysmems;

      // which other FBs we have peer access to
      // std::set<Memory> peer_fbs;

      // streams for different copy types and a pile for actual tasks
      // DPUStream *host_to_device_stream;
      // DPUStream *device_to_host_stream;
      // DPUStream *device_to_device_stream;
      // std::vector<DPUStream *> device_to_device_streams;
      // std::vector<DPUStream *> peer_to_peer_streams; // indexed by target
      // std::vector<DPUStream *> task_streams;
      atomic<unsigned> next_task_stream, next_d2d_stream;

      DPUEventPool event_pool;

      struct UpmemIpcMapping {
        NodeID owner;
        Memory mem;
        uintptr_t local_base;
        uintptr_t address_offset; // add to convert from original to local base
      };
      std::vector<UpmemIpcMapping> upmemipc_mappings;
      // std::map<NodeID, DPUStream *> upmemipc_streams;
      // const UpmemIpcMapping *find_ipc_mapping(Memory mem) const;
    };

    class DPUProcessor : public Realm::LocalTaskProcessor {
    public:
      DPUProcessor(DPU *_dpu, Processor _me, Realm::CoreReservationSet &crs,
                   size_t _stack_size);
      virtual ~DPUProcessor(void);

    public:
      virtual bool register_task(Processor::TaskFuncID func_id, CodeDescriptor &codedesc,
                                 const ByteArrayRef &user_data);

      virtual void shutdown(void);

    protected:
      virtual void execute_task(Processor::TaskFuncID func_id,
                                const ByteArrayRef &task_args);

    public:
      static DPUProcessor *get_current_dpu_proc(void);

      //   void stream_wait_on_event(dpu_set_t stream, upmem_bin event);
      //   void stream_synchronize(dpu_set_t stream);
      void device_synchronize(void);

      void dpu_memcpy(void *dst, const void *src, size_t size, UpmemMemcpyKind kind);
      //   void dpu_memcpy_async(void *dst, const void *src, size_t size,
      // 		    UpmemMemcpyKind kind, dpu_set_t stream);
      void dpu_memset(void *dst, int value, size_t count);
      //   void dpu_memset_async(void *dst, int value, size_t count, dpu_set_t
      //   stream);

    public:
      DPU *dpu;

      // data needed for kernel launches
      struct LaunchConfig {
        uint16_t tasklets;
        uint16_t dpus;
        size_t mram;
        LaunchConfig(uint16_t tasklets, uint16_t dpus, size_t _mram);
      };
      struct CallConfig : public LaunchConfig {
        dpu_set_t stream;
        CallConfig(uint16_t tasklets, uint16_t dpus, size_t _mram, dpu_set_t _stream);
      };
      std::vector<CallConfig> launch_configs;
      std::vector<char> kernel_args;
      std::vector<CallConfig> call_configs;
      bool block_on_synchronize;
      ContextSynchronizer ctxsync;

    protected:
      Realm::CoreReservation *core_rsrv;

      struct DPUTaskTableEntry {
        Processor::TaskFuncPtr fnptr;
        // DPU::StreamAwareTaskFuncPtr stream_aware_fnptr;
        ByteArray user_data;
      };

      // we're not using the parent's task table, but we can use the mutex
      // RWLock task_table_mutex;
      std::map<Processor::TaskFuncID, DPUTaskTableEntry> dpu_task_table;
    };

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
    };

    class DPUStream {
    public:
      DPUStream(DPU *_dpu, DPUWorker *_worker, int rel_priority = 0);
      ~DPUStream(void);

      DPU *get_dpu(void) const;
      dpu_set_t get_stream(void) const;

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

      void add_event(upmem_bin event, DPUWorkFence *fence,
                     DPUCompletionNotification *notification = NULL,
                     DPUWorkStart *start = NULL);

      DPU *dpu;
      DPUWorker *worker;

      dpu_set_t stream;

      Mutex mutex;

#define USE_CQ
#ifdef USE_CQ
      Realm::CircularQueue<DPUMemcpy *> pending_copies;
#else
      std::deque<DPUMemcpy *> pending_copies;
#endif
      bool issuing_copies;

      struct PendingEvent {
        upmem_bin event;
        DPUWorkFence *fence;
        DPUWorkStart *start;
        DPUCompletionNotification *notification;
      };
#ifdef USE_CQ
      Realm::CircularQueue<PendingEvent> pending_events;
#else
      std::deque<PendingEvent> pending_events;
#endif
    };

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

      upmem_bin get_event(bool external = false);
      void return_event(upmem_bin e, bool external = false);

    protected:
      Mutex mutex;
      int batch_size, current_size, total_size, external_count;
      std::vector<upmem_bin> available_events;
    };

    // an interface for receiving completion notification for a DPU operation
    //  (right now, just copies)
    class DPUCompletionNotification {
    public:
      virtual ~DPUCompletionNotification(void) {}

      virtual void request_completed(void) = 0;
    };

    class DPUPreemptionWaiter : public DPUCompletionNotification {
    public:
      DPUPreemptionWaiter(DPU *dpu);
      virtual ~DPUPreemptionWaiter(void) {}

    public:
      virtual void request_completed(void);

    public:
      void preempt(void);

    private:
      DPU *const dpu;
      Event wait_event;
    };

    class DPUCompletionEvent : public DPUCompletionNotification {
    public:
      void request_completed(void);

      DPURequest *req;
    };

    class DPURequest : public Request {
    public:
      const void *src_base;
      void *dst_base;
      // off_t src_dpu_off, dst_dpu_off;
      DPU *dst_dpu;
      DPUCompletionEvent event;
    };

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
      static void upmem_callback(dpu_set_t stream, uint32_t rank_id, void *data);
    };

    class DPUWorkStart : public Realm::Operation::AsyncWorkItem {
    public:
      DPUWorkStart(Realm::Operation *op);

      virtual void request_cancellation(void) { return; };

      void enqueue_on_stream(DPUStream *stream);

      virtual void print(std::ostream &os) const;

      void mark_dpu_work_start();

    protected:
      static void upmem_start_callback(dpu_set_t stream, uint32_t rank_id, void *data);
    };

    // An abstract base class for all DPU memcpy operations
    class DPUMemcpy { //: public DPUJob {
    public:
      DPUMemcpy(DPU *_dpu, DPUMemcpyKind _kind);
      virtual ~DPUMemcpy(void) {}

    public:
      virtual void execute(DPUStream *stream) = 0;

    public:
      DPU *const dpu;

    protected:
      DPUMemcpyKind kind;
    };

    class DPUMemcpyFence : public DPUMemcpy {
    public:
      DPUMemcpyFence(DPU *_dpu, DPUMemcpyKind _kind, DPUWorkFence *_fence);

      virtual void execute(DPUStream *stream);

    protected:
      DPUWorkFence *fence;
    };

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
      std::vector<hipEvent_t>().swap(available_events);
    }

    hipEvent_t DPUEventPool::get_event(bool external)
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

    void DPUEventPool::return_event(hipEvent_t e, bool external)
    {
      AutoLock<> al(mutex);

      assert(current_size < total_size);

      if(external) {
        assert(external_count);
        external_count--;
      }

      available_events[current_size++] = e;
    }

    class DPUReplHeapListener : public ReplicatedHeap::Listener {
    public:
      DPUUReplHeapListener(UpmemModule *_module);

      virtual void chunk_created(void *base, size_t bytes);
      virtual void chunk_destroyed(void *base, size_t bytes);

    protected:
      UpmemModule *module;
    };

  } // namespace Upmem

} // namespace Realm