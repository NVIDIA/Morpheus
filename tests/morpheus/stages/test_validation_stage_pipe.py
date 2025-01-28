# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

import pytest

from _utils import assert_path_exists
from _utils import assert_results
from _utils import calc_error_val
from morpheus.pipeline import LinearPipeline
from morpheus.stages.input.in_memory_source_stage import InMemorySourceStage
from morpheus.stages.postprocess.validation_stage import ValidationStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage


@pytest.mark.use_cudf
@pytest.mark.parametrize("write_results", [True, False])
def test_file_rw_serialize_deserialize_pipe(tmp_path, config, filter_probs_df, write_results: bool):
    if write_results:
        results_file_name = os.path.join(tmp_path, 'results.json')
    else:
        results_file_name = None

    pipe = LinearPipeline(config)
    pipe.set_source(InMemorySourceStage(config, [filter_probs_df]))
    pipe.add_stage(DeserializeStage(config))
    val_stage = pipe.add_stage(
        ValidationStage(config, val_file_name=filter_probs_df.to_pandas(), results_file_name=results_file_name))
    pipe.run()

    if write_results:
        assert_path_exists(results_file_name)
        results = calc_error_val(results_file_name)
        assert results.diff_rows == 0
    else:
        assert_results(val_stage.get_results())
