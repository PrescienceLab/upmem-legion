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

namespace Realm {
  namespace Upmem {

    extern Logger log_upmem;

    Kernel::Kernel(void) {}

    Kernel::Kernel(const char *_bin, dpu_set_t *_stream)
      : bin(_bin)
      , stream(_stream)
    {}

    void Kernel::load(void)
    {
      UpmemModule *mod = get_runtime()->get_module<UpmemModule>("UpmemModule");
      this->stream = mod->get_task_upmem_stream();
      CHECK_UPMEM(dpu_load(*this->stream, this->bin, NULL));
    }

    void Kernel::launch(void *args[], const char *symbol_name, size_t arg_size)
    {

      dpu_set_t dpu_proc;

#ifdef DEBUG_REALM
      assert(arg_size % 8 == 0 && "args_size must be multiple of 8 bytes");
#endif

      DPU_FOREACH(*stream, dpu_proc) { CHECK_UPMEM(dpu_prepare_xfer(dpu_proc, args)); }

      CHECK_UPMEM(dpu_push_xfer(*stream, DPU_XFER_TO_DPU, symbol_name, 0, arg_size,
                                DPU_XFER_ASYNC));

#ifdef PRINT_UPMEM
      // printing is a blocking operation. we need to read buffer once available.
      CHECK_UPMEM(dpu_launch(*stream, DPU_SYNCHRONOUS));
      printf("Display DPU logs for %s\n", bin);
      DPU_FOREACH(*stream, dpu_proc) { DPU_ASSERT(dpu_log_read(dpu_proc, stdout)); }
#else
      CHECK_UPMEM(dpu_launch(*stream, DPU_ASYNCHRONOUS));
#endif
    }

  }; // namespace Upmem
  using Upmem::log_upmem;
} // namespace Realm