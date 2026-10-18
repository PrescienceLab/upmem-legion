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

#ifndef REALM_UPMEM_ACCESS_H
#define REALM_UPMEM_ACCESS_H

#include "realm/inst_layout.h"
#include "realm/upmem/upmem_layout.h"

#ifndef DPURT
#define DPURT
#include <dpu> // UPMEM rt syslib
#define CHECK_UPMEM(x) DPU_ASSERT(x)
#endif

// this is disgusting
#define PADDING(x)                                                                       \
private:                                                                                 \
  union {                                                                                \
    char padding[x];                                                                     \
    alignas(x) struct { } _; };                                                          \
  static_assert(sizeof(padding) >= sizeof(_), "Padding is too small");

namespace Realm {
  namespace Upmem {

    class REALM_PUBLIC_API Kernel {
    public:
      Kernel(const char *bin);
      Kernel(void);
      virtual ~Kernel(void) {}

    public:
      virtual void launch(void *args[], const char *symbol_name, size_t arg_size);
      virtual void load(void);

      // launches the kernel, waits for it, and copies each DPU's partial_size-byte
      //  __host partial_symbol into partials (DPU order, room for
      //  REALM_UPMEM_MAX_DPUS_PER_RANK entries); returns the number of DPUs
      size_t launch_and_gather(void *args[], const char *symbol_name, size_t arg_size,
                               const char *partial_symbol, void *partials,
                               size_t partial_size);

      // launch_and_gather of one REDOP::RHS per DPU, folded with REDOP
      template <typename REDOP>
      typename REDOP::RHS launch_reduce(void *args[], const char *symbol_name,
                                        size_t arg_size, const char *partial_symbol);

    protected:
      // caller holds the rank's xfer mutex
      void start(dpu_set_t *stream, void *args[], const char *symbol_name,
                 size_t arg_size);

      const char *bin;
      const char *symbol_name;
    }; // end class Kernel

    template <typename REDOP>
    inline typename REDOP::RHS Kernel::launch_reduce(void *args[],
                                                     const char *symbol_name,
                                                     size_t arg_size,
                                                     const char *partial_symbol)
    {
      typename REDOP::RHS partials[REALM_UPMEM_MAX_DPUS_PER_RANK];
      size_t n = launch_and_gather(args, symbol_name, arg_size, partial_symbol, partials,
                                   sizeof(typename REDOP::RHS));
      typename REDOP::RHS result = REDOP::identity;
      for(size_t i = 0; i < n; i++)
        REDOP::template fold<true>(result, partials[i]);
      return result;
    }

  }; // namespace Upmem
};   // namespace Realm

#endif