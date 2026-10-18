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

#ifndef REALM_UPMEM_API_H
#define REALM_UPMEM_API_H

#include "realm/realm_config.h"
#include "realm/memory.h"

namespace Realm {
  namespace Upmem {
    // row size (bytes) of a local rank memory's interleaved layout, 0 if m isn't one
    REALM_PUBLIC_API size_t get_mram_row_bytes(Memory m);
  }; // namespace Upmem
};   // namespace Realm

#endif
