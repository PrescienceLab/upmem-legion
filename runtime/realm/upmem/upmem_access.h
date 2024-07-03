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
#include <dpu>

namespace Realm {
namespace Upmem {
    REALM_PUBLIC_API void LaunchKernel(const char* bin, void* args[], size_t arg_size, dpu_set_t* stream);

}
} // namespace Realm

#endif