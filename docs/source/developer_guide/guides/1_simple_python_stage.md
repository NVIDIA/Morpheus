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

# Simple Python Stage

> **Note**: The code for this guide can be found in the `examples/developer_guide/1_simple_python_stage` directory of the Morpheus repository.

## Background

Morpheus makes use of the MRC graph-execution framework. Morpheus pipelines are built on top of MRC pipelines, which are comprised of collections of nodes and edges, called segments (think sub-graphs),  which can in turn be connected by ingress/egress ports. In many common cases, an MRC pipeline will consist of only a single segment. Our Morpheus stages interact with the MRC segment to define, build, and add nodes to the MRC graph; the stages themselves can be thought of as packaged units of work to be applied to data flowing through the pipeline. These work units comprising an individual Morpheus stage may consist of a single MRC node, a small collection of nodes, or an entire MRC sub-graph.

## The Pass Through Stage

To start, we will implement a single stage that could be included in a pipeline. For illustration, this stage will do nothing but take the input from the previous stage and forward it to the next stage. All Morpheus stages have several things in common, so while this doesn't do too much, it ends up being a good starting point for writing a new stage. From there, we can add our functionality as needed. Morpheus provides two ways of defining a stage, as a stand-alone function or as a class.

### Stand-alone Function

The stand-alone function approach is the simplest way to define a stage. The function should accept a single argument, which will be the input message, and return a single value, which will be the output message. The function should be decorated with the {py:func}`~morpheus.pipeline.stage_decorator.stage` decorator.

```python
import typing

from morpheus.pipeline.stage_decorator import stage


@stage
def pass_thru_stage(message: typing.Any) -> typing.Any:
    # Return the message for the next stage
    return message
```

When using the `stage` decorator it is required to use type annotations for the function parameters and return type, as this will be used by the stage as the accept and output types. In the above example the stage decorator will use the name of the function as the name of the stage. If we wanted to specify a different name for the stage we could do so by passing the name to the decorator as follows:

```python
@stage(name="pass-thru")
def pass_thru_stage(message: typing.Any) -> typing.Any:
    # Return the message for the next stage
    return message
```

By default, Morpheus stages are assumed to require a GPU. However since this stage doesn't perform any specific GPU operations. We can indicate that the stage does not require a GPU by passing a tuple of supported execution modes to the decorator as follows:
```python
import typing

from morpheus.config import ExecutionMode
from morpheus.pipeline.stage_decorator import stage


@stage(name="pass-thru", execution_modes=(ExecutionMode.GPU, ExecutionMode.CPU))
def pass_thru_stage(message: typing.Any) -> typing.Any:
    # Return the message for the next stage
    return message
```

We can then add our stage to a pipeline as follows:
```python
config = Config()
pipeline = LinearPipeline(config)
# ... rest of pipeline setup
pipeline.add_stage(pass_thru_stage(config))
```

It is also possible to provide additional keyword arguments to the function. Consider the following example:
```python
@stage
def multiplier(message: MessageMeta, *, column: str, value: int | float = 2.0) -> MessageMeta:
    with message.mutable_dataframe() as df:
        df[column] = df[column] * value

    return message

# ...

pipe.add_stage(multiplier(config, column='probs', value=5))
```

### Stage Class

The class based approach to defining a stage offers a bit more flexibility, specifically the ability to validate constructor arguments, and perform any needed setup prior to being invoked in a pipeline. Defining this stage requires us to specify the stage type. Morpheus stages which contain a single input and a single output typically inherit from {py:class}`~morpheus.pipeline.single_port_stage.SinglePortStage`.  Stages that act as sources of data, in that they do not take an input from a prior stage but rather produce data from a source such as a file, Kafka service, or other external sources, will need to inherit from the {py:class}`~morpheus.pipeline.single_output_source.SingleOutputSource` base class.

Stages in Morpheus define what types of data they accept, and the type of data that they emit. In this example we are emitting messages of the same type that is received, this is actually quite common and Morpheus provides a mixin class, {py:class}`~morpheus.pipeline.pass_thru_type_mixin.PassThruTypeMixin`, to simplify this.

Similar to the function based stage, the class based stage will be not require a GPU, and we will indicate that it is able to be used in both GPU and CPU execution modes by utilizing the {py:class}`~morpheus.pipeline.execution_mode_mixins.GpuAndCpuMixin`.

Optionally, stages can be registered as a command with the Morpheus CLI using the {py:func}`~morpheus.cli.register_stage.register_stage` decorator. This allows for pipelines to be constructed from both pre-built stages and custom user stages via the command line. Any constructor arguments will be introspected using [`numpydoc`](https://numpydoc.readthedocs.io/en/latest/) and exposed as command line flags. Similarly, the class's docstrings will be exposed in the help string of the stage on the command line.

We start our class definition with a few basic imports:

```python
import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage


@register_stage("pass-thru")
class PassThruStage(PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):
```

There are four methods that need to be defined in our new subclass to implement the stage interface: `name`, `accepted_types`, `compute_schema`, `supports_cpp_node`, and `_build_single`. In practice, it is often necessary to define at least one more method which will perform the actual work of the stage; by convention, this method is typically named `on_data`, which we will define in our examples.

`name` is a property method; it should return a user-friendly name for the stage. Currently, this property is only used for debugging purposes, and there are no requirements on the content or format of the name. However by convention the string returned by this method should be the same as the string passed to the `register_stage` decorator.
```python
    @property
    def name(self) -> str:
        return "pass-thru"
```

The `accepted_types` method returns a tuple of message classes that this stage is able to accept as input. Morpheus uses this to validate that the parent of this stage emits a message that this stage can accept. Since our stage is a pass through, we will declare that we can accept any incoming message type. Note that production stages will often declare only a single Morpheus message class such as {py:class}`~morpheus.messages.MessageMeta` or {py:class}`~morpheus.messages.ControlMessage` (refer to the message classes defined in {py:mod}`~morpheus.messages` for a complete list).
```python
    def accepted_types(self) -> tuple:
        return (typing.Any,)
```

As mentioned previously we are making use of the `PassThruTypeMixin`, which defines the `compute_schema` method for us. This method returns the schema of the output message type. The `PassThruTypeMixin`, should be used anytime a stage receives and emits messages of the same type, even if it only accepts messages of a specific type and modifies the data, the data type remains the same. Had we not used the `PassThruTypeMixin`, we would have defined the `compute_schema` method as follows:
```python
from morpheus.pipeline.stage_schema import StageSchema
```
```python
    def compute_schema(self, schema: StageSchema):
        schema.output_schema.set_type(schema.input_type)
```

While the `compute_schema` method is simple enough to write, the real value of the `PassThruTypeMixin` presents itself for stages which can handle inputs from multiple upstream ports and emit messages on multiple output ports. However for now we are dealing with single port stages which are the most common type.

The `supports_cpp_node` method returns a boolean indicating if the stage has a C++ implementation. Since our example only contains a Python implementation we will return `False` here.
```python
    def supports_cpp_node(self) -> bool:
        return False
```

Our `on_data` method accepts an incoming message and returns a message. The returned message can be the same message instance that we received as our input or it could be a new message instance. The method is named `on_data` by convention; however, it is not part of the API. In the next section, we will register it as a callback in Morpheus.
```python
    def on_data(self, message: typing.Any):
        # Return the message for the next stage
        return message
```

Finally, the `_build_single` method will be used at stage build time to construct our node and wire it into the pipeline. `_build_single` receives an instance of an MRC segment builder (`mrc.Builder`) along with an `input_node` of type `mrc.SegmentObject`. We will be using the builder instance to construct a node from our stage and connecting it to the Morpheus pipeline. The return value of `_build_single` is our newly constructed node allowing downstream nodes to attach to our node.
```python
    def _build_single(self, builder: mrc.Builder,
                      input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
```

For our purposes, a Morpheus _stage_ defines the input data type the stage will accept, the unit of work to be performed on that data, and the output data type. In contrast each individual node or nodes comprising a _stage_'s unit of work are wired into the underlying MRC execution pipeline. To build the node, we will call the `make_node` method of the builder instance, passing it our `unique_name` property method and applying the map operator to the `on_data` method. We used the `unique_name` property, which will take the `name` property which we already defined and append a unique id to it.
```python
node = builder.make_node(self.unique_name, ops.map(self.on_data))
```

Next, we will define an edge connecting our new node to our parent node:
```python
builder.make_edge(input_node, node)
```

Finally, we will return our node.
```python
return node
```

## Putting the Stage Together
```python
import typing

import mrc
from mrc.core import operators as ops

from morpheus.cli.register_stage import register_stage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage


@register_stage("pass-thru")
class PassThruStage(PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):
    """
    A Simple Pass Through Stage
    """

    @property
    def name(self) -> str:
        return "pass-thru"

    def accepted_types(self) -> tuple:
        return (typing.Any, )

    def supports_cpp_node(self) -> bool:
        return False

    def on_data(self, message: typing.Any) -> typing.Any:
        # Return the message for the next stage
        return message

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:
        node = builder.make_node(self.unique_name, ops.map(self.on_data))
        builder.make_edge(input_node, node)

        return node
```

## Testing our new Stage
To start testing both our new function-based and class-based stages, we are going to construct a simple pipeline and add both stages to it. This pipeline will do the minimum work necessary to verify our pass through stages. Data will flow through our simple pipeline as follows:
1. A source stage will produce data and emit it into the pipeline.
1. This data will be read and processed by our pass through stage, in this case simply forwarding on the data.
1. A monitoring stage will record the messages from our pass through stage and terminate the pipeline.

First we will need to import a few things from Morpheus for this example to work. Note that this test script, which we will name `"run.py"`, assumes that we saved the code for the class based `PassThruStage` in a file named `"pass_thru.py"` and the function based `pass_thru_stage` named "pass_thru_deco.py" in the same directory.

```python
import logging
import os

from pass_thru import PassThruStage
from pass_thru_deco import pass_thru_stage

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.utils.logger import configure_logging
```


Before constructing the pipeline, we need to do a bit of environment configuration, starting with the Morpheus logger:
```python
configure_logging(log_level=logging.DEBUG)
```
We use the default configuration with the `DEBUG` logging level. The logger will output to both the console and a file. The logging handlers are non-blocking since they utilize a queue to send the log messages on a separate thread.

We can also use `configure_logging` to add one or more logging handlers to the default configuration. The added handlers will also be non-blocking. The following is from the Grafana example (`examples/digital_fingerprinting/production/grafana/README.md`) where we add a [Loki](https://grafana.com/oss/loki/) logging handler to also publish Morpheus logs to a Loki log aggregation server.
```python
loki_handler = logging_loki.LokiHandler(
    url=f"{loki_url}/loki/api/v1/push",
    tags={"app": "morpheus"},
    version="1",
)

configure_logging(loki_handler, log_level=log_level)
```

Next, we will build a Morpheus `Config` object. We will cover setting some common configuration parameters in the next guide. For now, it is important to know that we will always need to build a `Config` object:
```python
config = Config()
```

In this example, we will use the `FileSourceStage` class to read a large file in which each line is a JSON object that represents an email message. The stage will take these lines and package them as Morpheus message objects for our pass through stage to consume. Let's setup our source stage:
```python
pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False))
```

Next, we will add both versions of our new stage to the pipeline as well as a `MonitorStage` instance for each to measure the throughput of our stages:

```python
# Add the decorated function stage
pipeline.add_stage(pass_thru_stage(config))

# Add monitor to record the performance of the function based stage
pipeline.add_stage(MonitorStage(config))

# Add the class based stage
pipeline.add_stage(PassThruStage(config))

# Add monitor to record the performance of the class based stage
pipeline.add_stage(MonitorStage(config))
```

Finally, we run the pipeline:
```python
pipeline.run()
```

The output should display:
```
====Pipeline Pre-build====
====Pre-Building Segment: linear_segment_0====
====Pre-Building Segment Complete!====
====Pipeline Pre-build Complete!====
====Registering Pipeline====
====Building Pipeline====
====Building Pipeline Complete!====
====Registering Pipeline Complete!====
====Starting Pipeline====
====Pipeline Started====
====Building Segment: linear_segment_0====
Added source: <from-file-0; FileSourceStage(filename=examples/data/email_with_addresses.jsonlines, iterative=False, file_type=FileTypes.Auto, repeat=1, filter_null=True, parser_kwargs=None)>
  └─> morpheus.MessageMeta
Added stage: <pass_thru_stage-1; WrappedFunctionStage(on_data_fn=<function pass_thru_stage at 0x7f001f72bd00>, on_data_args=(), accept_type=None, return_type=None, needed_columns=None, on_data_kwargs={})>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
Added stage: <monitor-2; MonitorStage(description=Progress, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None, log_level=LogLevels.INFO)>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
Added stage: <pass-thru-3; PassThruStage()>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
Added stage: <monitor-4; MonitorStage(description=Progress, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None, log_level=LogLevels.INFO)>
  └─ morpheus.MessageMeta -> morpheus.MessageMeta
====Building Segment Complete!====
Progress[Complete]: 100 messages [00:01, 69.97 messages/s]
Progress[Complete]: 100 messages [00:01, 69.76 messages/s]
====Pipeline Complete====
```

## Putting the Pipeline Together
Note that this code assumes the `MORPHEUS_ROOT` environment variable is set to the root of the Morpheus project repository:
```python
import logging
import os

from pass_thru import PassThruStage
from pass_thru_deco import pass_thru_stage

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.utils.logger import configure_logging


def run_pipeline():
    # Enable the Morpheus logger
    configure_logging(log_level=logging.DEBUG)

    root_dir = os.environ['MORPHEUS_ROOT']
    input_file = os.path.join(root_dir, 'examples/data/email_with_addresses.jsonlines')

    config = Config()

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    pipeline.set_source(FileSourceStage(config, filename=input_file, iterative=False))

    # Add the decorated function stage
    pipeline.add_stage(pass_thru_stage(config))

    # Add monitor to record the performance of the function based stage
    pipeline.add_stage(MonitorStage(config))

    # Add the class based stage
    pipeline.add_stage(PassThruStage(config))

    # Add monitor to record the performance of the class based stage
    pipeline.add_stage(MonitorStage(config))

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
```


### Alternate Morpheus CLI example
The above example makes use of the Morpheus Python API. Alternately, we could test the class-based stage in a pipeline constructed using the Morpheus command line tool. We will need to pass in the path to our stage via the `--plugin` argument so that it will be visible to the command line tool.

> **Note**: For now, registering a stage with the CLI tool is currently only available to class based stages.

From the root of the Morpheus repo, run:
```bash
morpheus --log_level=debug --plugin examples/developer_guide/1_simple_python_stage/pass_thru.py \
  run pipeline-other \
  from-file --filename=examples/data/email_with_addresses.jsonlines \
  pass-thru \
  monitor
```
