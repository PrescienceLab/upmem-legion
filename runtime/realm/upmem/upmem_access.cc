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

#include "realm/upmem/upmem_access.h"
#include "realm/upmem/upmem_module.h"
#include "realm/upmem/upmem_internal.h"

namespace Realm {
  namespace Upmem {

    extern Logger log_upmem;

    Kernel::Kernel(void) {}

    Kernel::Kernel(const char *_bin)
      : bin(_bin)
    {}

    void Kernel::load(void) // top level task will call this
    {
      UpmemModule *mod = get_runtime()->get_module<UpmemModule>("upmem");
      assert(mod != NULL && "Kernel::load Upmem Module NULL");
      const char *rt_dir = getenv("LG_RT_DIR");
      assert(rt_dir != NULL && "Kernel::load needs LG_RT_DIR");
      std::string reset_path =
          std::string(rt_dir) + "/realm/upmem/upmem_mram_reset.up.o";

      for(std::vector<DPU *>::iterator it = mod->dpus.begin(); it != mod->dpus.end();
          it++) {
        DPU *dpu = *it;
        dpu_set_t *stream = dpu->stream->get_stream();

        // clear the heap from prior kernel invocations
        CHECK_UPMEM(dpu_load(*stream, reset_path.c_str(), NULL));
        CHECK_UPMEM(dpu_launch(*stream, DPU_SYNCHRONOUS));

        // once heap is cleared, load this
        CHECK_UPMEM(dpu_load(*stream, this->bin, NULL));

        // tell each DPU where it sits in the rank layout
        AutoLock<> al(dpu->get_xfer_mutex());
        std::vector<uint32_t> ids(dpu->nr_dpus);
        for(uint32_t i = 0; i < dpu->nr_dpus; i++) {
          ids[i] = i;
          CHECK_UPMEM(dpu_prepare_xfer(dpu->dpu_handles[i], &ids[i]));
        }
        CHECK_UPMEM(dpu_push_xfer(*stream, DPU_XFER_TO_DPU, "REALM_UPMEM_DPU_ID", 0,
                                  sizeof(uint32_t), DPU_XFER_DEFAULT));
        uint32_t shape[2] = {dpu->nr_dpus, dpu->granule};
        CHECK_UPMEM(dpu_broadcast_to(*stream, "REALM_UPMEM_SHAPE", 0, shape,
                                     sizeof(shape), DPU_XFER_DEFAULT));
      }
    }

    void Kernel::start(dpu_set_t *stream, void *args[], const char *symbol_name,
                       size_t arg_size)
    {
#ifdef DEBUG_REALM
      assert(arg_size % 8 == 0 && "args_size must be multiple of 8 bytes");
#endif
      // see DPU::copy_partial_word - a sync transfer must not queue behind async work
      CHECK_UPMEM(dpu_sync(*stream));
      CHECK_UPMEM(dpu_broadcast_to(*stream, symbol_name, 0, args, arg_size,
                                   DPU_XFER_DEFAULT));
      CHECK_UPMEM(dpu_launch(*stream, DPU_ASYNCHRONOUS));
    }

    void Kernel::launch(void *args[], const char *symbol_name, size_t arg_size)
    {
      UpmemModule *mod = get_runtime()->get_module<UpmemModule>("upmem");
      DPU *dpu = mod->get_task_dpu();
      assert(dpu && "Kernel::launch must be called from a DPU task");
      dpu_set_t *stream = dpu->stream->get_stream();

      // the UPMEM runtime isn't safe for concurrent calls on one rank
      AutoLock<> al(dpu->get_xfer_mutex());
      start(stream, args, symbol_name, arg_size);
#ifdef PRINT_UPMEM
      // printing is a blocking operation. we need to read buffer once available.
      CHECK_UPMEM(dpu_sync(*stream));
      dpu_set_t dpu_proc;
      DPU_FOREACH(*stream, dpu_proc) { DPU_ASSERT(dpu_log_read(dpu_proc, stdout)); }
#endif
    }

    size_t Kernel::launch_and_gather(void *args[], const char *symbol_name,
                                     size_t arg_size, const char *partial_symbol,
                                     void *partials, size_t partial_size)
    {
      UpmemModule *mod = get_runtime()->get_module<UpmemModule>("upmem");
      DPU *dpu = mod->get_task_dpu();
      assert(dpu && "Kernel::launch_and_gather must be called from a DPU task");
      dpu_set_t *stream = dpu->stream->get_stream();

      {
        AutoLock<> al(dpu->get_xfer_mutex());
        start(stream, args, symbol_name, arg_size);
        // queued right behind the launch, before any later kernel can overwrite it
        for(uint32_t i = 0; i < dpu->nr_dpus; i++)
          CHECK_UPMEM(dpu_prepare_xfer(dpu->dpu_handles[i],
                                       static_cast<char *>(partials) + i * partial_size));
        CHECK_UPMEM(dpu_push_xfer(*stream, DPU_XFER_FROM_DPU, partial_symbol, 0,
                                  partial_size, DPU_XFER_ASYNC));
      }

      // wait without blocking the processor
      DPUPreemptionWaiter waiter(dpu);
      dpu->stream->add_notification(&waiter);
      waiter.preempt();

      return dpu->nr_dpus;
    }

  }; // namespace Upmem
  using Upmem::log_upmem;
} // namespace Realm