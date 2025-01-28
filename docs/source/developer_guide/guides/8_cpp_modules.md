<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# C++ Morpheus Modules

## Background

See [Simple Python Module](./7_python_modules.md) for an introduction to Morpheus modules.

## The Pass-through Module

The following example will create a simple C++ module that passes through the input data without modification. This module will be written in C++ and would be compiled into the Morpheus core library.

**Note**: One thing that is different with respect to C++ modules, is that they are assumed to be stateless by default, meaning that the module itself can be released after the initialize function as been called. If you need a module whose state is persisted across the lifetime of the pipeline, you will also need to inherit from the `PersistentModule` class, which will cause the pipeline to hold a reference to the module until the pipeline is destroyed.

`my_test_module.hpp`

```c++
#pragma once
#include <morpheus/export.h>
#include <mrc/modules/properties/persistent.hpp>
#include <mrc/modules/segment_modules.hpp>
#include <nlohmann/json.hpp>

namespace morpheus {

class MORPHEUS_EXPORT MyTestModule: public mrc::modules::SegmentModule, public mrc::modules::PersistentModule
{
    using type_t = MyTestModule;

  public:
    DataLoaderModule(std::string module_name);
    DataLoaderModule(std::string module_name, nlohmann::json _config);

  protected:
    void initialize(mrc::segment::Builder& builder) override;
    std::string module_type_name() const override;

  private:
    int my_persistent_value{0};
};

}  // namespace morpheus
```

`my_test_module.cpp`

```c++
#include <mrc/modules/segment_modules.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/utils/type_utils.hpp>
#include <nlohmann/json.hpp>

#include <string>

namespace morpheus {
MyTestModule::MyTestModule(std::string module_name) : SegmentModule(module_name) {}

MyTestModule::MyTestModule(std::string module_name, nlohmann::json _config) :
  SegmentModule(std::move(module_name), std::move(_config))
{}

void MyTestModule::initialize(mrc::segment::Builder& builder) {
    auto passthrough_node = builder.make_node<std::shared_ptr<MyDataType>>("passthrough_node",
            rxcpp::operators::map([this](std::shared_ptr<MyDataType> data) {
                return data;
            }));

    register_input_port("input_0", passthrough_node);
    register_output_port("output_0", passthrough_node);
}

std::string MyTestModule::module_type_name() const
{
    return std::string(::mrc::boost_type_name<type_t>());
}
}
```

`my_test_module_registration.cpp`

```c++
#include "mrc/modules/module_registry.hpp"

#include "my_test_module.hpp"

int main(int argc, char** argv) {
    const std::vector<unsigned int> release_version = {1, 0, 0};

    auto module_constructor = [](std::string module_name, nlohmann::json config) {
        return std::make_shared<morpheus::MyTestModule>(module_name, config);
    };

    ModuleRegistry::register_module("my_test_module", "my_module_namespace", release_version, module_constructor);
}
```

## Advanced Topics

### Dynamic Module Creation and Loading

```c++
#include "mrc/version.hpp"
#include "mrc/modules/module_registry.hpp"

#include "my_test_module.hpp"

extern "C" {

const std::vector<unsigned int> TestModuleVersion{mrc_VERSION_MAJOR, mrc_VERSION_MINOR, mrc_VERSION_PATCH};

const char* MODULES[] = {"my_test_module::my_module_namespace"};

 bool MRC_MODULE_entrypoint_load()  // NOLINT
{
    using namespace mrc::modules;

    try
    {
        ModuleRegistry::register_module(
            "my_test_module",
            "my_module_namespace"
            TestModuleVersion,
            [](std::string module_name, nlohmann::json config) {
                return std::make_shared<mrc::modules::MyTestModule>(std::move(module_name), std::move(config));
            });
    } catch (...)
    {
        return false;
    }

    return true;
}

bool MRC_MODULE_entrypoint_unload()  // NOLINT
{
    using namespace mrc::modules;

    try
    {
        ModuleRegistry::unregister_module("my_test_module", "my_module_namespace");
    } catch (...)
    {
        return false;
    }

    return true;
}

unsigned int MRC_MODULE_entrypoint_list(const char** result)  // NOLINT
{
    *result = (const char*)(&MODULES);

    return 1; // Number of modules
}

```

The above code is an example of how to declare a shared module that can be loaded at runtime. If we assume this snippet is compile into `my_test_module.so`, we can load dynamically load the module at runtime using the following:

```c++
#include "mrc/modules/module_registry.hpp"

std::string get_modules_path() {
    return std::string{YOUR_MODULES_PATH};
}

int main(int argc, char** argv) {
    using namespace mrc::modules;

    auto plugin = PluginModule::create_or_acquire("my_test_module.so");
    plugin->set_library_directory(get_modules_path());
    plugin->load();

    std::string module_namespace{"my_module_namespace"};
    std::string module_name{"my_test_module"};

    ModuleRegistry::contains_namespace(module_namespace); // Should be true
    ModuleRegistry::contains(module_name, module_namespace); // Should be true
}
```
