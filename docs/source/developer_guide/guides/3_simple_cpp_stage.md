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

# Simple C++ Stage
## Building the Example
The code for this guide can be found in the `examples/developer_guide/3_simple_cpp_stage` directory of the Morpheus repository. There are two ways to build the example. The first is to build the examples along with Morpheus by passing the `-DMORPHEUS_BUILD_EXAMPLES=ON` flag to CMake, for users using the `scripts/compile.sh` at the root of the Morpheus repo can do this by setting the `CMAKE_CONFIGURE_EXTRA_ARGS` environment variable:
```bash
CMAKE_CONFIGURE_EXTRA_ARGS="-DMORPHEUS_BUILD_EXAMPLES=ON" ./scripts/compile.sh
```

The second method is to build the example as a standalone project. From the root of the Morpheus repo execute:
```bash
cd examples/developer_guide/3_simple_cpp_stage
./compile.sh

# Optionally install the package into the current python environment
pip install ./
```

## Overview
Morpheus offers the choice of writing pipeline stages in either Python or C++. For many use cases, a Python stage is perfectly fine. However, in the event that a Python stage becomes a bottleneck for the pipeline, then writing a C++ implementation for the stage becomes advantageous. The C++ implementations of Morpheus stages and messages utilize the [pybind11](https://pybind11.readthedocs.io/en/stable/index.html) library to provide Python bindings.

We have been defining our stages in Python up to this point, the option of defining a C++ implementation is only available to stages implemented as classes. Many of the stages included with Morpheus have both a Python and a C++ implementation, and Morpheus will use the C++ implementations by default when running in the GPU execution mode. When running in the CPU execution mode, Morpheus will always use the Python implementation.

If a stage does not have a C++ implementation, Morpheus will fall back to the Python implementation without any additional configuration. Morpheus stages which only contain a C++ implementation, still require a Python class to register the stage, and provide the stage's configuration.

In addition to C++ accelerated stage implementations, Morpheus also provides a C++ implementation for message primitives. When using the GPU GPU execution mode (the default), constructing one of the Python message classes defined under {py:mod}`~morpheus.messages` will return a Python object with bindings to the underlying C++ implementation.

Since we are defining our stages in Python, it becomes the responsibility of the Python stage to build a C++ accelerated node. This happens in the `_build_source` and `_build_single` methods. The Python stage should call `self._build_cpp_node()` to determine if a C++ node should be built, and ultimately it is the decision of a Python stage to build a Python node or a C++ node. It is perfectly acceptable to build a Python node when `self._build_cpp_node()` is returns `True`. It is not acceptable, however, to build a C++ node when `self._build_cpp_node()` returns `False`. The reason is the C++ implementations of Morpheus messages can be consumed by Python and C++ stage implementations alike. However the Python implementations of Morpheus messages cannot be consumed by the C++ implementations of stages.

Python stages which have a C++ implementation must advertise this functionality by returning a value of `True` from the `supports_cpp_node` method:

```python
def supports_cpp_node(self):
    return True
```

C++ message object declarations can be found in the header files that are located in the `morpheus/_lib/include/morpheus/messages` directory. For example, the `MessageMeta` class declaration is located in `morpheus/_lib/include/morpheus/messages/meta.hpp`. Since this code is outside of the `morpheus` directory it would be included as:

```cpp
#include <morpheus/messages/meta.hpp>
```

Morpheus C++ source stages inherit from the `PythonSource` class from MRC:

```cpp
template <typename OutputT, typename ContextT = mrc::runnable::Context>
class PythonSource  : ...
```

The `OutputT` type will be the datatype emitted by this stage. In contrast, general stages and sinks must inherit from the `PythonNode` class from MRC, which specifies both receive and emit types:

```cpp
template <typename InputT, typename OutputT, typename ContextT = mrc::runnable::Context>
class PythonNode : ...
```

Both the `PythonSource` and `PythonNode` classes are defined in the `pymrc/node.hpp` header.

> **Note**: `InputT` and `OutputT` types are typically `shared_ptr`s to a Morpheus message type. For example, `std::shared_ptr<MessageMeta>`. This allows the reference counting mechanisms used in Python and C++ to share the same count, properly cleaning up the objects when they are no longer referenced.

> **Note**: The "Python" in the `PythonSource` & `PythonNode` class names refers to the fact that these classes read and write objects registered with Python, not the implementation language.

## A Simple Pass Through Stage

As in our Python guide, we will start with a simple pass through stage which can be used as a starting point for future development of other stages. Note that by convention, C++ classes in Morpheus have the same name as their corresponding Python classes and are located under a directory named `_lib`. We will be following that convention. To start, we will create a `_lib` directory and a new empty `__init__.py` file.

While our Python implementation accepts messages of any type (in the form of Python objects), on the C++ side we don't have that flexibility since our node is subject to C++ static typing rules. In practice, this isn't a limitation as we usually know which specific message types we need to work with. For this example we will be working with the `ControlMessage` as our input and output type. This means that at build time our Python stage implementation is able to build a C++ node when the incoming type is `ControlMessage`, while falling back to the existing Python implementation otherwise.

To start with, we have our Morpheus and MRC-specific includes:

```cpp
#include <morpheus/export.h>            // for exporting symbols
#include <morpheus/messages/control.hpp>  // for ControlMessage
#include <mrc/segment/builder.hpp>      // for Segment Builder
#include <mrc/segment/object.hpp>       // for Segment Object
#include <pymrc/node.hpp>               // for PythonNode
#include <rxcpp/rx.hpp>
```

We'll want to define our stage in its own namespace. In this case, we will name it `morpheus_example`, giving us a namespace and class definition like:

```cpp
namespace morpheus_example {

using namespace morpheus;

// pybind11 sets visibility to hidden by default; we want to export our symbols
class MORPHEUS_EXPORT PassThruStage
  : public mrc::pymrc::PythonNode<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>;
    using base_t::sink_type_t;
    using base_t::source_type_t;
    using base_t::subscribe_fn_t;

    PassThruStage();

    subscribe_fn_t build_operator();
};
```

We explicitly set the visibility for the stage object to default by importing:
```cpp
#include <morpheus/export.h>
```
Then adding `MORPHEUS_EXPORT`, which is defined in `/build/autogenerated/include/morpheus/export.h` and is compiler agnostic, to the definition of the stage object.
This is due to a pybind11 requirement for module implementations to default symbol visibility to hidden (`-fvisibility=hidden`). More details about this can be found in the [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes).
Any object, struct, or function that is intended to be exported should have `MORPHEUS_EXPORT` included in the definition.

For simplicity, we defined `base_t` as an alias for our base class type because the definition can be quite long. Our base class type also defines a few additional type aliases for us: `subscribe_fn_t`, `sink_type_t` and `source_type_t`. The `sink_type_t` and `source_type_t` aliases are shortcuts for the sink and source types that this stage will be reading and writing. In this case both the `sink_type_t` and `source_type_t` resolve to `std::shared_ptr<ControlMessage>`. `subscribe_fn_t` (read as "subscribe function type") is an alias for:

```cpp
std::function<rxcpp::subscription(rxcpp::observable<InputT>, rxcpp::subscriber<OutputT>)>
```

This means that an MRC subscribe function accepts an `rxcpp::observable` of type `InputT` and `rxcpp::subscriber` of type `OutputT` and returns a subscription. In our case, both `InputT` and `OutputT` are `std::shared_ptr<ControlMessage>`.

All Morpheus C++ stages receive an instance of an MRC Segment Builder and a name (Typically this is the Python class' `unique_name` property) when constructed from Python. Note that C++ stages don't receive an instance of the Morpheus `Config` object. Therefore, if there are any attributes in the `Config` needed by the C++ class, it is the responsibility of the Python class to extract them and pass them in as parameters to the C++ class.

We will also define an interface proxy object to keep the class definition separated from the Python interface. This isn't strictly required, but it is a convention used internally by Morpheus. Our proxy object will define a static method named `init` which is responsible for constructing a `PassThruStage` instance and returning it wrapped in a `shared_ptr`. There are many common Python types that pybind11 [automatically converts](https://pybind11.readthedocs.io/en/latest/advanced/cast/overview.html#conversion-table) to their associated C++ types. The MRC `Builder` is a C++ object with Python bindings. However there are other instances such as checking for values of `None` where the casting from Python to C++ types is not automatic. The proxy interface object fulfills this need and is used to help insulate Python bindings from internal implementation details.

```cpp
struct MORPHEUS_EXPORT PassThruStageInterfaceProxy
{
    static std::shared_ptr<mrc::segment::Object<PassThruStage>> init(mrc::segment::Builder &builder,
                                                                     const std::string &name);
};
```

### The Complete C++ Stage Header

```cpp
#pragma once

#include <morpheus/export.h>            // for exporting symbols
#include <morpheus/messages/control.hpp>  // for ControlMessage
#include <mrc/segment/builder.hpp>      // for Segment Builder
#include <mrc/segment/object.hpp>       // for Segment Object
#include <pymrc/node.hpp>               // for PythonNode
#include <rxcpp/rx.hpp>

#include <memory>
#include <string>
#include <thread>

// IWYU pragma: no_include "morpheus/objects/data_table.hpp"
// IWYU pragma: no_include <boost/fiber/context.hpp>

namespace morpheus_example {

using namespace morpheus;

// pybind11 sets visibility to hidden by default; we want to export our symbols
class MORPHEUS_EXPORT PassThruStage
  : public mrc::pymrc::PythonNode<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<ControlMessage>, std::shared_ptr<ControlMessage>>;
    using base_t::sink_type_t;
    using base_t::source_type_t;
    using base_t::subscribe_fn_t;

    PassThruStage();

    subscribe_fn_t build_operator();
};

struct MORPHEUS_EXPORT PassThruStageInterfaceProxy
{
    static std::shared_ptr<mrc::segment::Object<PassThruStage>> init(mrc::segment::Builder& builder,
                                                                     const std::string& name);
};

}  // namespace morpheus_example
```

### Source Code Definition

Our includes section:

```cpp
#include "pass_thru.hpp"

#include <pybind11/pybind11.h>
#include <pymrc/utils.hpp>  // for pymrc::import

#include <exception>
```

The constructor for our class is responsible for passing the output of `build_operator` to our base class, as well as calling the constructor for `PythonNode`:

```cpp
PassThruStage::PassThruStage() : PythonNode(base_t::op_factory_from_sub_fn(build_operator())) {}
```

Note that the output of `build_operator()` is not passed directly to the `PythonNode` constructor and instead gets passed to `base_t::op_factory_from_sub_fn()`. This is because reactive operators can be defined two ways:

1. Using the short form `std::function<rxcpp::observable<T>(rxcpp::observable<R>)` which is good when you can use an existing `rxcpp` operator
2. Using the long form `std::function<rxcpp::subscription(rxcpp::observable<T>, rxcpp::subscriber<R>)>` which allows for more customization and better control over the lifetime of objects.

It's possible to convert between the two signatures which is exactly what `base_t::op_factory_from_sub_fn()` does. If you wanted to use the short form, you could define the constructor of `PassThruStage` using:

```cpp
PassThruStage::PassThruStage() :
  PythonNode([](rxcpp::observable<sink_type_t> obs){ return obs; })
{}
```

However, this doesn't illustrate well how to customize a stage. For this reason, we will be using the long form signature for our examples.

The `build_operator` method defines an observer which is subscribed to our input `rxcpp::observable`. The observer consists of three functions that are typically lambdas:  `on_next`, `on_error`, and `on_completed`. Typically, these three functions call the associated methods on the output subscriber.

```cpp
PassThruStage::subscribe_fn_t PassThruStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(
            rxcpp::make_observer<sink_type_t>([this, &output](sink_type_t x) { output.on_next(std::move(x)); },
                                              [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                                              [&]() { output.on_completed(); }));
    };
}
```

Note the use of `std::move` in the `on_next` function. In Morpheus, our messages often contain both large payloads as well as Python objects where performing a copy necessitates acquiring the Python [Global Interpreter Lock (GIL)](https://docs.python.org/3.10/glossary.html#term-global-interpreter-lock). In either case, unnecessary copies can become a performance bottleneck, and much care is taken to limit the number of copies required for data to move through the pipeline.

There are situations in which a C++ stage does need to interact with Python, and therefore acquiring the GIL is a requirement. This is typically accomplished using pybind11's [`gil_scoped_acquire`](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil) RAII class inside of a code block. Conversely there are situations in which we want to ensure that we are not holding the GIL and in these situations pybind11's [`gil_scoped_release`](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil) class can be used.

For stages it is important to ensure that the GIL is released before calling the output's `on_next` method. Consider the following `on_next` lambda function:

```cpp
using namespace pybind11::literals;
pybind11::gil_scoped_release no_gil;

[this, &output](sink_type_t msg) {
    auto mutable_info = msg->meta->get_mutable_info();

    std::shared_ptr<MessageMeta> new_meta;
    {
        // Acquire the GIL
        pybind11::gil_scoped_acquire gil;
        auto df = mutable_info.checkout_obj();

        // Maka a copy of the original DataFrame
        auto copied_df = df.attr("copy")("deep"_a = true);

        // Now that we are done with `df` return it to the owner
        mutable_info.return_obj(std::move(df));

        // do something with copied_df
        new_meta = MessageMeta::create_from_python(std::move(copied_df));
    }  // GIL is released

    output.on_next(std::move(new_meta));
}
```

We scoped the acquisition of the GIL such that it is held only for the parts of the code where it is strictly necessary. In the above example, when we exit the code block, the `gil` variable will go out of scope and release the global interpreter lock.

### Python Proxy and Interface

The things that all proxy interfaces need to do are:
1. Construct the stage using the `mrc::segment::Builder::construct_object` method
2. Return a `shared_ptr` to the stage wrapped in a `mrc::segment::Object`

```cpp
std::shared_ptr<mrc::segment::Object<PassThruStage>> PassThruStageInterfaceProxy::init(mrc::segment::Builder& builder,
                                                                                       const std::string& name)
{
    return builder.construct_object<PassThruStage>(name);
}
```

The Python interface itself defines a Python module named `morpheus_example` and a Python class in that module named `PassThruStage`. Note that the only method we are exposing to Python is the interface proxy's `init` method. The class will be exposed to Python code as `lib_.morpheus_example.PassThruStage`.

```cpp
namespace py = pybind11;

// Define the pybind11 module m.
PYBIND11_MODULE(pass_thru_cpp, m)
{
    mrc::pymrc::import(m, "morpheus._lib.messages");

    py::class_<mrc::segment::Object<PassThruStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PassThruStage>>>(m, "PassThruStage", py::multiple_inheritance())
        .def(py::init<>(&PassThruStageInterfaceProxy::init), py::arg("builder"), py::arg("name"));
}
```

### The Complete C++ Stage Implementation

```cpp
#include "pass_thru.hpp"

#include <pybind11/pybind11.h>
#include <pymrc/utils.hpp>  // for pymrc::import

#include <exception>

namespace morpheus_example {

PassThruStage::PassThruStage() : PythonNode(base_t::op_factory_from_sub_fn(build_operator())) {}

PassThruStage::subscribe_fn_t PassThruStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(
            rxcpp::make_observer<sink_type_t>([this, &output](sink_type_t x) { output.on_next(std::move(x)); },
                                              [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                                              [&]() { output.on_completed(); }));
    };
}

std::shared_ptr<mrc::segment::Object<PassThruStage>> PassThruStageInterfaceProxy::init(mrc::segment::Builder& builder,
                                                                                       const std::string& name)
{
    return builder.construct_object<PassThruStage>(name);
}

namespace py = pybind11;

// Define the pybind11 module m.
PYBIND11_MODULE(pass_thru_cpp, m)
{
    mrc::pymrc::import(m, "morpheus._lib.messages");

    py::class_<mrc::segment::Object<PassThruStage>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<PassThruStage>>>(m, "PassThruStage", py::multiple_inheritance())
        .def(py::init<>(&PassThruStageInterfaceProxy::init), py::arg("builder"), py::arg("name"));
}

}  // namespace morpheus_example
```

### Python Changes

We need to make a few minor adjustments to our Python implementation of the `PassThruStage`. First, we will need to change the return value of the `supports_cpp_node` method to indicate that our stage now supports a C++ implementation.
```python
def supports_cpp_node(self):
   return True
```

Next, as mentioned previously, our Python implementation can support messages of any type, however the C++ implementation can only support instances of `ControlMessage`. To do this we will override the `compute_schema` method to store the input type.
```python
def compute_schema(self, schema: StageSchema):
    super().compute_schema(schema)  # Call PassThruTypeMixin's compute_schema method
    self._input_type = schema.input_type
```
> **Note**: We are still using the `PassThruTypeMixin` to handle the requirements of setting the output type.

As mentioned in the previous section, our `_build_single` method needs to be updated to build a C++ node when the input type is `ControlMessage` and when `self._build_cpp_node()` returns `True`.

```python
def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
    if self._build_cpp_node() and isinstance(self._input_type, ControlMessage):
        from ._lib import pass_thru_cpp

        node = pass_thru_cpp.PassThruStage(builder, self.unique_name)
    else:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))

    builder.make_edge(input_node, node)
    return node
```
> **Note**: We lazily imported the C++ module to avoid importing it when it is not needed.


## Putting the Stage Together
```python
import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stage_schema import StageSchema


@register_stage("pass-thru")
class PassThruStage(PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):

    def __init__(self, config: Config):
        super().__init__(config)
        self._input_type = None

    @property
    def name(self) -> str:
        return "pass-thru"

    def accepted_types(self) -> tuple:
        return (typing.Any, )

    def supports_cpp_node(self) -> bool:
        return True

    def compute_schema(self, schema: StageSchema):
        super().compute_schema(schema)  # Call PassThruTypeMixin's compute_schema method
        self._input_type = schema.input_type

    def on_data(self, message: typing.Any):
        # Return the message for the next stage
        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        if self._build_cpp_node() and isinstance(self._input_type, ControlMessage):
            from ._lib import pass_thru_cpp

            node = pass_thru_cpp.PassThruStage(builder, self.unique_name)
        else:
            node = builder.make_node(self.unique_name, ops.map(self.on_data))

        builder.make_edge(input_node, node)
        return node
```

## Testing the Stage
To test the updated stage we will build a simple pipeline using the Morpheus command line tool. In order to illustrate the stage building a C++ node only when the input type is a `ControlMessage` we will insert the `pass-thru` stage in twice in the pipeline. In the first instance the input type will be `MessageMeta` and the stage will fallback to using a Python node, and in the second instance the input type will be a `ControlMessage` and the stage will build a C++ node.

```bash
PYTHONPATH="examples/developer_guide/3_simple_cpp_stage/src" \
morpheus --log_level=debug --plugin "simple_cpp_stage.pass_thru" \
    run pipeline-other \
    from-file --filename=examples/data/email_with_addresses.jsonlines \
    pass-thru \
    monitor \
    deserialize \
    pass-thru \
    monitor
```
