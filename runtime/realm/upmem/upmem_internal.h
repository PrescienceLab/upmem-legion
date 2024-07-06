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

#include "realm/upmem/upmem_module.h"

#include "realm/upmem/upmem_events.h"
#include "realm/upmem/upmem_memory.h"
#include "realm/upmem/upmem_proc.h"
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
    class DPUReplHeapListener;
    // memory.h
    class DPUMRAMMemory;
    // events.h
    class DPUEventPool;
    // module.h
    class UpmemModule;

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

    // class ContextSynchronizer {
    // public:
    //   ContextSynchronizer(DPU *_dpu, int _device_id, CoreReservationSet &crs,
    //                       int _max_threads);
    //   ~ContextSynchronizer();

    //   void add_fence(DPUWorkFence *fence);

    //   void shutdown_threads();

    //   void thread_main();

    // protected:
    //   DPU *dpu;

    //   int device_id;
    //   int max_threads;
    //   Mutex mutex;
    //   Mutex::CondVar condvar;
    //   bool shutdown_flag;
    //   DPUWorkFence::FenceList fences;
    //   int total_threads, sleeping_threads, syncing_threads;
    //   std::vector<Thread *> worker_threads;
    //   CoreReservation *core_rsrv;
    // }; // end class ContextSynchronizer

    class DPUReplHeapListener : public ReplicatedHeap::Listener {
    public:
      DPUReplHeapListener(UpmemModule *_module);

      virtual void chunk_created(void *base, size_t bytes);
      virtual void chunk_destroyed(void *base, size_t bytes);

    protected:
      UpmemModule *module;
    }; // end class DPUReplHeapListener

    namespace ThreadLocal {
      static REALM_THREAD_LOCAL DPUProcessor *current_dpu_proc = 0;
      static REALM_THREAD_LOCAL DPUStream *current_dpu_stream = 0;
      static REALM_THREAD_LOCAL std::set<DPUStream *> *created_dpu_streams = 0;
      static REALM_THREAD_LOCAL int context_sync_required = 0;
    }; // namespace ThreadLocal

  }; // namespace Upmem

}; // namespace Realm

#endif