# Copyright (c) 2024-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import pathlib
import sys
import typing

import click

from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import parse_log_level
from morpheus.config import Config
from morpheus.config import CppConfig
from morpheus.config import ExecutionMode
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import stage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.general.trigger_stage import TriggerStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.utils.logger import configure_logging

logger = logging.getLogger(f"morpheus.{__name__}")


@click.command()
@click.option('--use_cpu_only',
              default=False,
              type=bool,
              is_flag=True,
              help=("Whether or not to run in CPU only mode, setting this to True will disable C++ mode."))
@click.option("--log_level",
              default="DEBUG",
              type=click.Choice(get_log_levels(), case_sensitive=False),
              callback=parse_log_level,
              show_default=True,
              help="Specify the logging level to use.")
@click.option(
    "--in_file",
    help="Input file",
    required=True,
    type=click.Path(exists=True, readable=True),
)
@click.option(
    "--out_file",
    help="Output file",
    type=click.Path(dir_okay=False),
    default=".tmp/output/cpu_only_out.csv",
    required=True,
)
def run_pipeline(log_level: int, use_cpu_only: bool, in_file: pathlib.Path, out_file: pathlib.Path):
    # Enable the default logger
    configure_logging(log_level=log_level)

    if use_cpu_only:
        execution_mode = ExecutionMode.CPU
    else:
        execution_mode = ExecutionMode.GPU

    config = Config()
    config.execution_mode = execution_mode

    pipeline = LinearPipeline(config)

    pipeline.set_source(FileSourceStage(config, filename=in_file))

    pipeline.add_stage(MonitorStage(config, description="source"))

    pipeline.add_stage(TriggerStage(config))

    @stage(execution_modes=(execution_mode, ))
    def print_msg(msg: typing.Any) -> typing.Any:
        log_msg = [f"Receive a message of type {type(msg)}"]
        if isinstance(msg, MessageMeta):
            log_msg.append(f"- df type: {type(msg.df)}")

        logger.debug(" ".join(log_msg))

        return msg

    pipeline.add_stage(print_msg(config))

    pipeline.add_stage(DeserializeStage(config))

    pipeline.add_stage(MonitorStage(config, description="deserialize"))

    @stage(execution_modes=(execution_mode, ))
    def calculate_totals(msg: ControlMessage, *, total_column_name: str = "total") -> ControlMessage:
        meta = msg.payload()

        with meta.mutable_dataframe() as df:
            logger.debug("Received a ControlMessage with a dataframe of type %s", type(df))
            df[total_column_name] = df.select_dtypes(include="number").sum(axis=1)

        return msg

    pipeline.add_stage(calculate_totals(config))
    pipeline.add_stage(SerializeStage(config))
    pipeline.add_stage(WriteToFileStage(config, filename=out_file, overwrite=True))
    pipeline.build()

    logger.info("Running pipeline\tC++ mode = %s\texecution_mode = %s",
                CppConfig.get_should_use_cpp(),
                config.execution_mode)

    pipeline.run()

    known_gpu_packages = ['cudf', 'cuml', 'tensorrt', 'torch']
    known_gpu_packages_loaded = [pkg in sys.modules for pkg in known_gpu_packages]

    if any(known_gpu_packages_loaded):
        for (i, pkg) in enumerate(known_gpu_packages):
            if known_gpu_packages_loaded[i]:
                msg = f"{pkg} is loaded"
                if use_cpu_only:
                    logger.error(msg)
                else:
                    logger.info(msg)
    else:
        logger.info("No GPU packages loaded")


if __name__ == "__main__":
    run_pipeline()
