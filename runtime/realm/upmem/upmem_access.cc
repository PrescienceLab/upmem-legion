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
    REALM_PUBLIC_API void upmem()
    {
      // struct dpu_set_t dpus;
      // dpu_alloc(1, dpus, NULL);
      // dpu_sync(); // test

      Upmem::UpmemModule *mod = get_runtime()->get_module<Upmem::UpmemModule>("upmem");
      assert(mod);
    }
  }; // namespace Upmem
  using Upmem::log_upmem;
} // namespace Realm