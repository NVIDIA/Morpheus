# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import os

import click

from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import parse_log_level
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.messages import RawPacketMessage
from morpheus.pipeline.linear_pipeline import LinearPipeline
from morpheus.stages.doca.doca_convert_stage import DocaConvertStage
from morpheus.stages.doca.doca_source_stage import DocaSourceStage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.utils.logger import configure_logging


@click.command()
@click.option(
    "--nic_addr",
    help="NIC PCI Address",
    required=True,
)
@click.option(
    "--gpu_addr",
    help="GPU PCI Address",
    required=True,
)
@click.option(
    "--num_threads",
    default=len(os.sched_getaffinity(0)),
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of internal pipeline threads to use.",
)
@click.option(
    "--edge_buffer_size",
    default=1024 * 16,
    type=click.IntRange(min=1),
    show_default=True,
    help="Size of edge buffers.",
)
@click.option(
    "--max_batch_delay_sec",
    default=3.0,
    type=float,
    show_default=True,
    help="Maximum amount of time in seconds to buffer incoming packets.",
)
@click.option(
    "--buffer_channel_size",
    default=None,
    type=click.IntRange(min=2),
    show_default=True,
    help=("Size of the internal buffer channel used by the DocaConvertStage, if None, the value of `--edge_buffer_size`"
          " will be used."),
)
@click.option("--log_level",
              default="INFO",
              type=click.Choice(get_log_levels(), case_sensitive=False),
              callback=parse_log_level,
              show_default=True,
              help="Specify the logging level to use.")
@click.option("--output_file",
              default=None,
              help="File to output to, if not supplied, the to-file sink will be omitted.")
def run_pipeline(nic_addr: str,
                 gpu_addr: str,
                 num_threads: int,
                 edge_buffer_size: int,
                 max_batch_delay_sec: float,
                 buffer_channel_size: int,
                 log_level: int,
                 output_file: str | None):
    # Enable the default logger
    configure_logging(log_level=log_level)

    config = Config()
    config.mode = PipelineModes.NLP

    # Below properties are specified by the command line
    config.num_threads = num_threads
    config.edge_buffer_size = edge_buffer_size

    pipeline = LinearPipeline(config)

    # add doca source stage
    pipeline.set_source(DocaSourceStage(config, nic_addr, gpu_addr, 'udp'))

    def count_raw_packets(message: RawPacketMessage):
        return message.num

    pipeline.add_stage(
        MonitorStage(config,
                     description="DOCA GPUNetIO Raw rate",
                     unit='pkts',
                     determine_count_fn=count_raw_packets,
                     delayed_start=True))

    pipeline.add_stage(
        DocaConvertStage(config, max_batch_delay_sec=max_batch_delay_sec, buffer_channel_size=buffer_channel_size))

    pipeline.add_stage(MonitorStage(config, description="Convert rate", unit='pkts', delayed_start=True))

    if output_file is not None:
        pipeline.add_stage(WriteToFileStage(config, filename=output_file, overwrite=True))

    # Build the pipeline here to see types in the vizualization
    pipeline.build()

    # Run the pipeline
    pipeline.run()


if __name__ == "__main__":
    run_pipeline()
