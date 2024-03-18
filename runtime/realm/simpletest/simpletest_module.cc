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

#include "realm/simpletest/simpletest_module.h"


#include <stdio.h> 

namespace Realm {
  Logger log_simpletest("simpletest");


  namespace SimpleTest {
    ////////////////////////////////////////////////////////////////////////
    //
    // class SimpleTestModuleConfig

    SimpleTestModuleConfig::SimpleTestModuleConfig(void)
      : ModuleConfig("simpletest")
    {
    }

    void SimpleTestModuleConfig::configure_from_cmdline(std::vector<std::string>& cmdline)
    {
      // read command line parameters
      CommandLineParser cp;

      cp.add_option_int("-ll:test_nm", cfg_test_nm);

      bool ok = cp.parse_command_line(cmdline);
      if(!ok) {
        log_simpletest.fatal() << "error reading SimpleTest command line parameters";
        assert(false);
      }
    }
    ////////////////////////////////////////////////////////////////////////
    //
    // class SimpleTestModule

    SimpleTestModule::SimpleTestModule(void)
      : Module("simpletest")
      , config(nullptr)
    {
    }
      
    SimpleTestModule::~SimpleTestModule(void)
    {
      assert(config != nullptr);
      config = nullptr;
    }

    /*static*/ ModuleConfig *SimpleTestModule::create_module_config(RuntimeImpl *runtime)
    {
      SimpleTestModuleConfig *config = new SimpleTestModuleConfig();
      return config;
    }

    /*static*/ Module *SimpleTestModule::create_module(RuntimeImpl *runtime)
    {
      // create a module to fill in with stuff 
      SimpleTestModule *m = new SimpleTestModule;

      SimpleTestModuleConfig *config =
          checked_cast<SimpleTestModuleConfig *>(runtime->get_module_config("simpletest"));
      assert(config != nullptr);
      assert(config->finish_configured);
      assert(m->name == config->get_name());
      assert(m->config == nullptr);
      m->config = config;

      return m;
    }

    // do any general initialization - this is called after all configuration is
    //  complete
    void SimpleTestModule::initialize(RuntimeImpl *runtime)
    {
      Module::initialize(runtime);
      printf("Hello wolrd\n");
    }

    // create any memories provided by this module (default == do nothing)
    //  (each new MemoryImpl should use a Memory from RuntimeImpl::next_local_memory_id)
    void SimpleTestModule::create_memories(RuntimeImpl *runtime)
    {
      Module::create_memories(runtime);
    }

    // create any processors provided by the module (default == do nothing)
    //  (each new ProcessorImpl should use a Processor from
    //   RuntimeImpl::next_local_processor_id)
    void SimpleTestModule::create_processors(RuntimeImpl *runtime)
    {
      Module::create_processors(runtime);
    }
    
    // create any DMA channels provided by the module (default == do nothing)
    void SimpleTestModule::create_dma_channels(RuntimeImpl *runtime)
    {
      Module::create_dma_channels(runtime);
    }

    // create any code translators provided by the module (default == do nothing)
    void SimpleTestModule::create_code_translators(RuntimeImpl *runtime)
    {
      Module::create_code_translators(runtime);
    }

    // clean up any common resources created by the module - this will be called
    //  after all memories/processors/etc. have been shut down and destroyed
    void SimpleTestModule::cleanup(void)
    {
      Module::cleanup();
    }

  }; // namespace SimpleTest

}; // namespace Realm
