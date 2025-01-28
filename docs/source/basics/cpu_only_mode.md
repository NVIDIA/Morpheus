<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Morpheus CPU-Only Mode
By default, Morpheus is designed to take advantage of the GPU for accelerated processing. However, there are cases where it may be necessary to run Morpheus on a system without access to a GPU. To address this need, Morpheus provides a CPU-only execution mode. Many stages within Morpheus require a GPU to run, while others can operate in both GPU and CPU execution mode. Attempting to add a GPU-only stage to a pipeline that is configured to operate in CPU-only mode will result in an error.

## Execution Modes
By default, Morpheus will run in GPU execution mode. Users have the choice of specifying the execution mode with either the Python API or from the command line.

### Python API
Execution modes are defined in the `morpheus.config.ExecutionMode` enumeration, which is then specified in the `execution_mode` attribute of the `morpheus.config.Config` object. The following example demonstrates how to set the execution mode of a pipeline to CPU-only:

```python
from morpheus.config import Config
from morpheus.config import ExecutionMode
```

```python
config = Config()
config.execution_mode = ExecutionMode.CPU
```

> **Note**: The `execution_mode` and all other attributes of the `morpheus.config.Config` object must be set prior to constructing either the pipeline or any stage objects. The first time a `Config` object is used to construct a pipeline or stage, the `Config` object will freeze the configuration and prevent any further changes.

#### Examples
The `examples/cpu_only`, `examples/developer_guide/1_simple_python_stage` and `examples/developer_guide/2_2_rabbitmq` examples demonstrate pipelines that are able to operate in both GPU and CPU execution modes.

### Command Line
The `--use_cpu_only` flag is available as an option to the `morpheus run` sub-command.

```bash
morpheus run --use_cpu_only pipeline-other --help
```

#### Example
The following is a simple command line example of a pipeline that can execute in CPU-only mode. To begin, ensure that you have fetched the examples dataset by running the following command from the root of the Morpheus repository:
```bash
./scripts/fetch_data.py fetch examples
```

Then, run the following command to run the pipeline:
```bash
morpheus --log_level=INFO \
  run --use_cpu_only pipeline-other \
  from-file --filename=examples/data/email_with_addresses.jsonlines \
  deserialize \
  monitor \
  serialize \
  to-file --filename=.tmp/output.jsonlines --overwrite
```

## Designing Stages for CPU Execution
It is up to the author of each stage to decide which execution modes are supported. Options are: CPU, GPU, or both. As mentioned previously, the default execution mode is GPU; authors of stages which require a GPU do not need to make any changes to their stage definitions.

### DataFrames and Tensors
The selection of the execution mode implies selection of DataFrame and tensor types. In GPU mode, Morpheus will use [cuDF](https://docs.rapids.ai/api/cudf/stable/) DataFrames and tensors are represented as [CuPy](https://cupy.dev/) `ndarray` objects. In CPU mode, Morpheus will use [pandas](https://pandas.pydata.org/) DataFrames and [NumPy](https://numpy.org/) `ndarray` objects.

|Mode|DataFrame|Tensor|
| -- | ------- | ---- |
|GPU|[cuDF](https://docs.rapids.ai/api/cudf/stable/)|[CuPy](https://cupy.dev/)|
|CPU|[pandas](https://pandas.pydata.org/)|[NumPy](https://numpy.org/)|

### Stages Defined with `@stage` and `@source` Decorators
Both the `@stage` and `@source` decorators have an optional `execution_modes` parameter that accepts a tuple of `morpheus.config.ExecutionMode` values, which is used to specify the supported execution modes of the stage.

#### CPU-only Source & Stage Examples
```python
import logging

import pandas as pd

from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import MessageMeta
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import source
from morpheus.pipeline.stage_decorator import stage
from morpheus.utils.logger import configure_logging

logger = logging.getLogger(f"morpheus.{__name__}")

@source(execution_modes=(ExecutionMode.CPU, ))
def simple_source(num_rows: int = 10) -> MessageMeta:
    df = pd.DataFrame({"a": range(num_rows)})
    message = MessageMeta(df)
    yield message

@stage(execution_modes=(ExecutionMode.CPU, ))
def print_msg(msg: MessageMeta) -> MessageMeta:
    logger.info(f"Receive a message with a DataFrame of type: {type(msg.df)}")
    return msg

def main():
    configure_logging(log_level=logging.INFO)
    pipeline = LinearPipeline(config)
    pipeline.set_source(simple_source(config))
    pipeline.add_stage(print_msg(config))
    pipeline.run()

if __name__ == "__main__":
    main()
```

#### CPU and GPU Source and Stage Examples
Supporting both CPU and GPU execution modes requires writing code that can handle both types of DataFrames and `ndarray` objects. In many cases, code designed to work with pandas will work with cuDF, and code designed to work with NumPy will work with CuPy, without requiring any changes to the code. However, in some cases, the API may differ slightly and there is a need to know the payload type. Care must be taken not to directly import `cudf` or any other package requiring a GPU when running in CPU mode on a system without a GPU. Morpheus provides some helper methods to assist with this in the {py:mod}`~morpheus.utils.type_utils` module, such as {py:func}`~morpheus.utils.type_utils.is_cudf_type`, {py:func}`~morpheus.utils.type_utils.get_df_class`, and {py:func}`~morpheus.utils.type_utils.get_array_pkg`.

With a few simple modifications, the previous example now supports both CPU and GPU execution modes. The `get_df_class` function is used to determine the DataFrame type to use, and we added a command line flag to switch between the two execution modes.

```python
import logging

import click

from morpheus.config import Config
from morpheus.config import ExecutionMode
from morpheus.messages import MessageMeta
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import source
from morpheus.pipeline.stage_decorator import stage
from morpheus.utils.logger import configure_logging
from morpheus.utils.type_utils import get_df_class

logger = logging.getLogger(f"morpheus.{__name__}")


@source(execution_modes=(ExecutionMode.GPU, ExecutionMode.CPU))
def simple_source(num_rows: int = 10) -> MessageMeta:
    df_class = get_df_class()  # Returns either cudf.DataFrame or pandas.DataFrame
    df = df_class({"a": range(num_rows)})
    message = MessageMeta(df)
    yield message


@stage(execution_modes=(ExecutionMode.GPU, ExecutionMode.CPU))
def print_msg(msg: MessageMeta) -> MessageMeta:
    logger.info(f"Receive a message with a DataFrame of type: {type(msg.df)}")
    return msg


@click.command()
@click.option('--use_cpu_only',
              default=False,
              type=bool,
              is_flag=True,
              help=("Whether or not to run in CPU only mode, setting this to True will disable C++ mode."))
def main(use_cpu_only: bool):
    configure_logging(log_level=logging.INFO)

    if use_cpu_only:
        execution_mode = ExecutionMode.CPU
    else:
        execution_mode = ExecutionMode.GPU

    config = Config()
    config.execution_mode = execution_mode
    pipeline = LinearPipeline(config)
    pipeline.set_source(simple_source(config))
    pipeline.add_stage(print_msg(config))
    pipeline.run()


if __name__ == "__main__":
    main()
```

### Source and Stage Classes
Similar to the `@source` and `@stage` decorators, class-based sources and stages can also be defined to advertise which execution modes they support. The base class for all source and stage classes, `StageBase`, defines a `supported_execution_modes` method for this purpose, which can be overridden in a derived class. The method in the base class is defined as:

```python
def supported_execution_modes(self) -> tuple[ExecutionMode]:
    return (ExecutionMode.GPU, )
```

Stage authors are free to inspect constructor arguments of the stage to determine which execution modes are supported. However, for many stages the supported execution modes do not change based upon the constructor arguments. In these cases the {py:class}`~morpheus.pipeline.execution_mode_mixins.GpuAndCpuMixin` and {py:class}`~morpheus.pipeline.execution_mode_mixins.CpuOnlyMixin` mixins can be used to simplify the implementation.

Example class definition:
```python
from morpheus.cli.register_stage import register_stage
from morpheus.pipeline.execution_mode_mixins import GpuAndCpuMixin
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage


@register_stage("pass-thru")
class PassThruStage(PassThruTypeMixin, GpuAndCpuMixin, SinglePortStage):
    ...
```

#### GpuAndCpuMixin
In the previous decorators example, we discussed utilizing various helper methods available in the {py:mod}`~morpheus.utils.type_utils` module to assist in writing code that is able to operate in both CPU and GPU execution modes. To simplify this further, the `GpuAndCpuMixin` mixin adds these helper methods to the class. At the time of this writing, they are:

- `df_type_str` - Returns either `"cudf"` or `"pandas"`.
- `get_df_pkg` - Returns either the `cudf` or `pandas` module.
- `get_df_class` - Returns either the `cudf.DataFrame` or `pandas.DataFrame` class.

### Stages with C++ Implementations
C++ stages have the ability to interact with cuDF DataFrames via the [libcudf](https://docs.rapids.ai/api/libcudf/stable/) library; however, no such C++ library exists for pandas DataFrames. As a result, any stages which contain both a Python and a C++ implementation, the Python implementation will be used in CPU mode, and the C++ implementation will be used in GPU mode. For these stages, the Python implementation is then free to assume DataFrames are of type `pandas.DataFrame` and tensors are of type `numpy.ndarray`.

A stage which contains only a C++ implementation will not be able to run in CPU mode.
