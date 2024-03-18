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

#ifndef REALM_UPMEM_MODULE_H
#define REALM_UPMEM_MODULE_H
#define MEGABYTE (2 << 20)
#define KILOBYTE (2 << 10)

#include "realm/realm_config.h"
#include "realm/module.h"
#include "realm/processor.h"
#include "realm/network.h"
#include "realm/atomics.h"

namespace Realm {
  namespace Upmem {
    class DPU;
    class DPUWorker;
    struct DPUInfo;
    // class DPUZCMemory;
    class DPUReplHeapListener;

    class UpmemModuleConfig : public ModuleConfig {
      friend class UpmemModule;

    protected:
      UpmemModuleConfig(void);

    public:
      virtual void configure_from_cmdline(std::vector<std::string> &cmdline);

    public:
      // configurations
      int cfg_num_dpus = 8;
      int cfg_tasklets = 16;
      // size_t cfg_zc_mem_size = 64 << 20, cfg_zc_ib_size = 256 << 20;
      // size_t cfg_fb_mem_size = 256 << 20, cfg_fb_ib_size = 128 << 20;
      // bool cfg_use_dynamic_fb = true;
      // size_t cfg_dynfb_max_size = ~size_t(0);
      // std::string cfg_dpu_idxs;
      unsigned cfg_task_streams = 12, cfg_d2d_streams = 4;
      bool cfg_use_worker_threads = false, cfg_use_shared_worker = true,
           cfg_pin_sysmem = true;
      bool cfg_fences_use_callbacks = false;
      bool cfg_suppress_hijack_warning = false;
      unsigned cfg_skip_dpu_count = 0;
      bool cfg_skip_busy_dpus = false;
      size_t cfg_min_avail_mem = 0;
      int cfg_task_context_sync = -1; // 0 = no, 1 = yes, -1 = default (based on hijack)
      int cfg_max_ctxsync_threads = 4;
      bool cfg_multithread_dma = false;
      size_t cfg_hostreg_limit = 1 << 30;
      int cfg_d2d_stream_priority = -1;
      bool cfg_use_upmem_ipc = true;

      // resources
      bool resource_discovered = false;
      int res_num_dpus = 0;
      // size_t res_min_fbmem_size = 0;
      // std::vector<size_t> res_fbmem_sizes;
    };

    // our interface to the rest of the runtime
    class UpmemModule : public Module {
    protected:
      UpmemModule(void);

    public:
      virtual ~UpmemModule(void);

      static ModuleConfig *create_module_config(RuntimeImpl *runtime);

      static Module *create_module(RuntimeImpl *runtime);

      // do any general initialization - this is called after all configuration is
      //  complete
      virtual void initialize(RuntimeImpl *runtime);

      // create any memories provided by this module (default == do nothing)
      //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
      virtual void create_memories(RuntimeImpl *runtime);

      // create any processors provided by the module (default == do nothing)
      //  (each new ProcessorImpl should use a Processor from
      //   RuntimeImpl::next_local_processor_id)
      virtual void create_processors(RuntimeImpl *runtime);

      // create any DMA channels provided by the module (default == do nothing)
      virtual void create_dma_channels(RuntimeImpl *runtime);

      // create any code translators provided by the module (default == do nothing)
      virtual void create_code_translators(RuntimeImpl *runtime);

      // clean up any common resources created by the module - this will be called
      //  after all memories/processors/etc. have been shut down and destroyed
      virtual void cleanup(void);

    public:
      UpmemModuleConfig *config;
      RuntimeImpl *runtime;

      // "global" variables live here too
      DPUWorker *shared_worker;
      std::map<DPU *, DPUWorker *> dedicated_workers;
      std::vector<DPUInfo *> dpu_info;
      std::vector<DPU *> dpus;
      // void *zcmem_cpu_base, *zcib_cpu_base;
      // DPUZCMemory *zcmem;
      std::vector<void *> registered_host_ptrs;
      DPUReplHeapListener *rh_listener;
    };

  }; // namespace Upmem

}; // namespace Realm

#endif
