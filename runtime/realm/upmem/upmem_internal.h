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
#ifndef REALM_UPMEM_INT_H
#define REALM_UPMEM_INT_H

#ifndef DPURT
#define DPURT
#include <dpu> // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

#define HERE() fprintf(stderr, "we are at line %d in file %s\n", __LINE__, __FILE__)

#include "realm/upmem/upmem_module.h"

#include "realm/upmem/upmem_events.h"
#include "realm/upmem/upmem_memory.h"
#include "realm/upmem/upmem_stream.h"
#include "realm/upmem/upmem_workers.h"
#include "realm/upmem/upmem_dma.h"

namespace Realm {
  namespace Upmem {

    typedef uint32_t upmemDevice_t;

    struct DPUInfo {
      int index;
      uint16_t size = 128;
      upmemDevice_t device;
      static const size_t MAX_NAME_LEN = 64;
      std::set<upmemDevice_t> peers;
    };

    // Forard declaration
    class DPU;
    class DPUProcessor;
    class DPUReplHeapListener;
    // memory.h
    class DPUMRAMMemory;
    // events.h
    class DPUEventPool;
    // module.h
    class UpmemModule;
    // workers.h
    class DPUWorkFence;

    extern UpmemModule *upmem_module_singleton;

    class UpmemDeviceMemoryInfo : public ModuleSpecificInfo {
    public:
      UpmemDeviceMemoryInfo(int _device_id);

      int device_id;
      DPU *dpu;
    };

    class DPU {
    public:
      DPU(UpmemModule *_module, DPUInfo *_info, DPUWorker *worker, int _device_id);

      DPU() { assert(0); } // should never happen

      DPU(const DPU &rhs) { assert(0); }
      DPU &operator=(const DPU &rhs)
      {
        assert(0);
        return *this;
      }

      ~DPU(void);

      void push_context(void);
      void pop_context(void);

      void create_processor(RuntimeImpl *runtime, size_t stack_size);
      void create_mram_memory(RuntimeImpl *runtime, size_t size);
      void create_dma_channels(RuntimeImpl *runtime);

    public:
      UpmemModule *module;
      DPUInfo *info;
      DPUWorker *worker;
      DPUProcessor *proc;
      DPUMRAMMemory *mram;
      DPUStream *stream;

      // upmemCtx_t context;
      int device_id;
      char *mram_base;

      // which system memories have been registered and can be used for cuMemcpyAsync
      std::set<Memory> pinned_sysmems;
      // which other mram we have peer access to
      std::set<Memory> peer_mram;

      DPUStream *find_stream(struct dpu_set_t *stream) const;
      DPUStream *get_null_task_stream(void) const;
      DPUStream *get_next_task_stream(bool create = false);

      std::vector<DPUStream *> task_streams;
      atomic<unsigned> next_task_stream;

      DPUEventPool event_pool;

      struct UpmemIpcMapping {
        NodeID owner;
        Memory mem;
        uintptr_t local_base;
        uintptr_t address_offset; // add to convert from original to local base
      };
      std::vector<UpmemIpcMapping> upmemipc_mappings;
      std::map<NodeID, DPUStream *> upmemipc_streams;
      const UpmemIpcMapping *find_ipc_mapping(Memory mem) const;

    }; // end class DPU

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

    class DPUReplHeapListener : public ReplicatedHeap::Listener {
    public:
      DPUReplHeapListener(UpmemModule *_module);

      virtual void chunk_created(void *base, size_t bytes);
      virtual void chunk_destroyed(void *base, size_t bytes);

    protected:
      UpmemModule *module;
    }; // end class DPUReplHeapListener

  }; // namespace Upmem

}; // namespace Realm

#endif