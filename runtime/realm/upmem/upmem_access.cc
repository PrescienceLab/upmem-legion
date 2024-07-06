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

namespace Realm {
  namespace Upmem {

    extern Logger log_upmem;
    REALM_PUBLIC_API void LaunchKernel(const char *bin, void *args[],
                                       const char *symbol_name, size_t arg_size,
                                       dpu_set_t *stream)
    {
      int i = 0;

      dpu_set_t dpu_proc;

      DPU_ASSERT(dpu_alloc(1, NULL, stream));
      printf("load: %s\n", bin);

      DPU_ASSERT(dpu_load(*stream, bin, NULL));
      printf("debug me: arg_size = %ld\n", arg_size);

      DPU_FOREACH(*stream, dpu_proc, i) { DPU_ASSERT(dpu_prepare_xfer(dpu_proc, args)); }

      DPU_ASSERT(dpu_push_xfer(*stream, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, arg_size,
                               DPU_XFER_DEFAULT));

      DPU_ASSERT(dpu_launch(*stream, DPU_ASYNCHRONOUS));
    }

  }; // namespace Upmem
  using Upmem::log_upmem;
} // namespace Realm