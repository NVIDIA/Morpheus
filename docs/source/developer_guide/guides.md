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

# Developer Guides

## Morpheus Stages

Morpheus includes a number of pre-defined stage implementations to choose from when building a custom
pipeline, each of which can be included and configured to suit your application.

- [List of available Morpheus stages](../stages/morpheus_stages.md)

There are likely going to be situations that require writing a custom stage. Morpheus stages are written in
Python and optionally may include a C++ implementation. The following guides outline how to create your own stages
in both Python and C++.

- [Simple Python Stage](./guides/1_simple_python_stage.md)
- [Real-World Application: Phishing Detection](./guides/2_real_world_phishing.md)
- [Simple C++ Stage](./guides/3_simple_cpp_stage.md)
- [Creating a C++ Source Stage](./guides/4_source_cpp_stage.md)

> **Note**: The code for the above guides can be found in the `examples/developer_guide` directory of the Morpheus repository. To build the C++ examples, pass `-DMORPHEUS_BUILD_EXAMPLES=ON` to CMake when building Morpheus. Users building Morpheus with the provided `scripts/compile.sh` script can do so by setting the `CMAKE_CONFIGURE_EXTRA_ARGS` environment variable:
> ```bash
> CMAKE_CONFIGURE_EXTRA_ARGS="-DMORPHEUS_BUILD_EXAMPLES=ON" ./scripts/compile.sh
> ```

## Morpheus Modules

Morpheus includes, as of version 23.03, a number of pre-defined module implementations to choose from when building a
custom pipeline. Modules can be thought of as units of work, which exist at a lower level than stages. Modules can
be defined, registered, chained, nested, and loaded at runtime. Modules can be written in Python or C++.

- [List of available Morpheus modules](../modules/index.md)

There are likely going to be situations that require writing a custom module, either for creating your own
reusable work units, or for creating a new compound module from a set of existing primitives. The following guides
will walk through the process of creating a custom module in Python and C++.

- [Python Modules](./guides/7_python_modules.md)
- [C++ Modules](./guides/8_cpp_modules.md)

## Morpheus Control messages

- [Control Messages Overview](./guides/9_control_messages.md)

## Example Workflows

- [Digital Fingerprinting (DFP)](./guides/5_digital_fingerprinting.md)
- [Digital Fingerprinting (DFP) Reference](./guides/6_digital_fingerprinting_reference.md)
- [Modular DFP Reference](./guides/10_modular_pipeline_digital_fingerprinting.md)
