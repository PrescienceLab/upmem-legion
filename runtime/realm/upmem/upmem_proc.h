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

#ifndef REALM_UPMEM_PROC_H
#define REALM_UPMEM_PROC_H

#include "realm/proc_impl.h"
#include "realm/processor.h"
#include "realm/threads.h"

#include "realm/upmem/upmem_memory.h"
#include "realm/upmem/upmem_workers.h"

namespace Realm {
  namespace Upmem {
    // forward declarations
    // internal.h
    class DPU;

// TODO: fix this eventually
// this is due to the cyclic headers
#ifndef DPUWORKFENCE
#define DPUWORKFENCE
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
#endif

    class ContextSynchronizer {
    public:
      ContextSynchronizer(DPU *_dpu, int _device_id, CoreReservationSet &crs,
                          int _max_threads);
      ~ContextSynchronizer();

      void add_fence(DPUWorkFence *fence);

      void shutdown_threads();

      void thread_main();

    protected:
      DPU *dpu;

      int device_id;
      int max_threads;
      Mutex mutex;
      Mutex::CondVar condvar;
      bool shutdown_flag;
      DPUWorkFence::FenceList fences;
      int total_threads, sleeping_threads, syncing_threads;
      std::vector<Thread *> worker_threads;
      CoreReservation *core_rsrv;
    }; // end class ContextSynchronizer

    typedef void (*StreamAwareTaskFuncPtr)(const void *args, size_t arglen,
                                           const void *user_data, size_t user_data_len,
                                           Processor proc, struct dpu_set_t *stream);

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

      //   void stream_wait_on_event(dpu_set_t stream, upmemEvent_t event);
      //   void stream_synchronize(dpu_set_t stream);
      void device_synchronize(void);

      void dpu_memcpy(void *dst, const void *src, size_t size, DPUMemcpyKind kind);
      void dpu_memcpy_async(void *dst, const void *src, size_t size, DPUMemcpyKind kind,
                            dpu_set_t stream);
      void dpu_memset(void *dst, int value, size_t count);
      void dpu_memset_async(void *dst, int value, size_t count, dpu_set_t stream);

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
        struct dpu_set_t stream;
        CallConfig(uint16_t tasklets, uint16_t dpus, size_t _mram,
                   struct dpu_set_t _stream);
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
        Upmem::StreamAwareTaskFuncPtr stream_aware_fnptr;
        ByteArray user_data;
      };

      // we're not using the parent's task table, but we can use the mutex
      // RWLock task_table_mutex;
      std::map<Processor::TaskFuncID, DPUTaskTableEntry> dpu_task_table;

    }; // end class DPUProcessor

    // we want to subclass the scheduler to replace the execute_task method, but we also
    // want to
    //  allow the use of user or kernel threads, so we apply a bit of template magic
    //  (which only works because the constructors for the KernelThreadTaskScheduler and
    //  UserThreadTaskScheduler classes have the same prototypes)

    template <typename T>
    class DPUTaskScheduler : public T {
    public:
      DPUTaskScheduler(Processor _proc, Realm::CoreReservation &_core_rsrv,
                       DPUProcessor *_dpu_proc);

      virtual ~DPUTaskScheduler(void);

    protected:
      virtual bool execute_task(Task *task);
      virtual void execute_internal_task(InternalTask *task);

      // might also need to override the thread-switching methods to keep TLS up to date

      DPUProcessor *dpu_proc;
    };

  }; // namespace Upmem
};   // namespace Realm

#endif
