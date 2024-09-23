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

#ifndef REALM_UPMEM_REDOP_H
#define REALM_UPMEM_REDOP_H

#include "realm/realm_config.h"

#if !defined(DEVICE_DPU_CODE) // only host code will include this
#ifndef DPURT
#define DPURT
#include <dpu> // UPMEM rt syslib
#if 0
#define CHECK_UPMEM(x)                                                                   \
  {                                                                                      \
    dpu_error_t _drc = x;                                                                \
    HERE();                                                                              \
    printf("upmem returns %d DPU_OK = %d " #x "\n", _drc, DPU_OK);                       \
    DPU_ASSERT(_drc);                                                                    \
  }
#else
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif
#endif
#endif

namespace Realm {
  namespace Upmem {
#if defined(DEVICE_DPU_CODE)
    namespace ReductionKernels {
      template <typename REDOP, bool EXCL>
      void apply_upmem_kernel(uintptr_t lhs_base, uintptr_t lhs_stride,
                              uintptr_t rhs_base, uintptr_t rhs_stride, size_t count,
                              REDOP redop)
      {
        size_t tid = me();
        for(size_t idx = tid; tid < count; tid += NR_TASKLETS)
          redop.template apply_upmem<EXCL>(
              *reinterpret_cast<typename REDOP::LHS *>(lhs_base + lhs_stride),
              *reinterpret_cast<const typename REDOP::RHS *>(rhs_base + rhs_stride));
      }

      template <typename REDOP, bool EXCL>
      void fold_upmem_kernel(uintptr_t rhs1_base, uintptr_t rhs1_stride,
                             uintptr_t rhs2_base, uintptr_t rhs2_stride, size_t count,
                             REDOP redop)
      {
        size_t tid = me();
        for(size_t idx = tid; tid < count; tid += NR_TASKLETS)
          redop.template fold_upmem<EXCL>(
              *reinterpret_cast<typename REDOP::RHS *>(rhs1_base + rhs1_stride),
              *reinterpret_cast<const typename REDOP::RHS *>(rhs2_base + rhs2_stride));
      }

    }; // namespace ReductionKernels
#endif
    // this helper adds the appropriate kernels for REDOP to a ReductionOpUntyped,
    //  although the latter is templated to work around circular include deps
    template <typename REDOP, typename T /*= ReductionOpUntyped*/>
    void add_upmem_redop_kernels(T *redop)
    {
      // store the host proxy function pointer, as it's the same for all
      //  devices - translation to actual cudaFunction_t's happens later
      // redop->upmem_apply_excl_fn = APPLY_UPMEM_KERNEL;
      // reinterpret_cast<void *>(&ReductionKernels::apply_upmem_kernel<REDOP, true>);
      // redop->upmem_fold_excl_fn = FOLD_UPMEM_KERNEL;
      // reinterpret_cast<void *>(&ReductionKernels::fold_upmem_kernel<REDOP, true>);
    }

  }; // namespace Upmem
};   // namespace Realm

#endif
