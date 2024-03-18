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

#ifndef REALM_SIMPLETEST_MODULE_H
#define REALM_SIMPLETEST_MODULE_H

#include "realm/realm_config.h"
#include "realm/module.h"
#include "realm/processor.h"
#include "realm/network.h"
#include "realm/atomics.h"
#include "realm/logging.h"
#include "realm/cmdline.h"
#include "realm/proc_impl.h"
#include "realm/threads.h"
#include "realm/runtime_impl.h"
#include "realm/utils.h"
#include "realm/tasks.h"
#include "realm/event_impl.h"
#include "realm/idx_impl.h"

namespace Realm {

  namespace SimpleTest {

    class SimpleTestModuleConfig : public ModuleConfig {
      friend class SimpleTestModule;
    protected:
      SimpleTestModuleConfig(void);

    public:
      virtual void configure_from_cmdline(std::vector<std::string>& cmdline);
    protected:
        int cfg_test_nm = 0; // test
    };

   // our interface to the rest of the runtime
    class SimpleTestModule : public Module {
    protected:
      SimpleTestModule(void);
      
    public:
      virtual ~SimpleTestModule(void);

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
      SimpleTestModuleConfig *config;
    };

  }; // namespace SimpleTest

}; // namespace Realm

#endif


