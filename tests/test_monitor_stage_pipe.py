#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Generator

import numpy as np
import pytest

import cudf

from _utils import assert_results
from morpheus.common import IndicatorsFontStyle
from morpheus.common import IndicatorsTextColor
from morpheus.messages import ControlMessage
from morpheus.messages import MessageMeta
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.stage_decorator import stage
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.input.in_memory_data_generation_stage import InMemoryDataGenStage
from morpheus.stages.output.compare_dataframe_stage import CompareDataFrameStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


def sample_message_meta_generator(df_rows: int, df_cols: int, count: int) -> Generator[MessageMeta, None, None]:
    data = {f'col_{i}': range(df_rows) for i in range(df_cols)}
    df = cudf.DataFrame(data)
    meta = MessageMeta(df)
    for _ in range(count):
        yield meta


@pytest.mark.gpu_mode
def test_cpp_monitor_stage_pipe(config):
    config.num_threads = 1

    df_rows = 10
    df_cols = 3
    expected_df = next(sample_message_meta_generator(df_rows, df_cols, 1)).copy_dataframe()

    count = 20

    cudf_generator = partial(sample_message_meta_generator, df_rows, df_cols, count)

    @stage
    def dummy_control_message_process_stage(msg: ControlMessage) -> ControlMessage:
        matrix_a = np.random.rand(3000, 3000)
        matrix_b = np.random.rand(3000, 3000)
        matrix_c = np.dot(matrix_a, matrix_b)
        msg.set_metadata("result", matrix_c[0][0])

        return msg

    # The default determine_count_fn for MessageMeta and ControlMessage returns the number of rows in the DataFrame
    # This customized_determine_count_fn returns 1 for each MessageMeta
    def customized_determine_count_fn(msg: MessageMeta) -> int:  # pylint: disable=unused-argument
        return 1

    pipe = LinearPipeline(config)
    pipe.set_source(InMemoryDataGenStage(config, cudf_generator, output_data_type=MessageMeta))
    pipe.add_stage(DeserializeStage(config, ensure_sliceable_index=True))
    pipe.add_stage(
        MonitorStage(config,
                     description="preprocess",
                     unit="records",
                     text_color=IndicatorsTextColor.green,
                     font_style=IndicatorsFontStyle.underline))
    pipe.add_stage(dummy_control_message_process_stage(config))
    pipe.add_stage(
        MonitorStage(config,
                     description="postprocess",
                     unit="records",
                     text_color=IndicatorsTextColor.blue,
                     font_style=IndicatorsFontStyle.italic))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(
        MonitorStage(config, description="sink", unit="MessageMeta", determine_count_fn=customized_determine_count_fn))
    comp_stage = pipe.add_stage(CompareDataFrameStage(config, expected_df))
    pipe.run()

    assert_results(comp_stage.get_results())
