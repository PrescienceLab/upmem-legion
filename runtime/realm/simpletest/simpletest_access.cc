
/* Copyright 2023 Stanford University, NVIDIA Corporation
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


#include "realm/simpletest/simpletest_access.h"

#include "realm/simpletest/simpletest_module.h"

namespace Realm {

  namespace SimpleTest {
    extern Logger log_simpletest;
    REALM_PUBLIC_API void simpletest() {
        SimpleTest::SimpleTestModule *mod = get_runtime()->get_module<SimpleTest::SimpleTestModule>("simpletest");
        assert(mod);
    }
  };
  using SimpleTest::log_simpletest;
}