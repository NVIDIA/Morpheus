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

from morpheus.cli import hookimpl
from morpheus.cli.stage_registry import LazyStageInfo
from morpheus.cli.stage_registry import StageRegistry
from morpheus.config import PipelineModes


@hookimpl
def morpheus_cli_collect_stages(registry: StageRegistry):

    registry.add_stage_info(
        LazyStageInfo("gnn-fraud-classification",
                      __package__ + ".stages.classification_stage.ClassificationStage",
                      modes=[PipelineModes.OTHER]))

    registry.add_stage_info(
        LazyStageInfo("fraud-graph-construction",
                      __package__ + ".stages.graph_construction_stage.FraudGraphConstructionStage",
                      modes=[PipelineModes.OTHER]))

    registry.add_stage_info(
        LazyStageInfo("gnn-fraud-sage",
                      __package__ + ".stages.graph_sage_stage.GraphSAGEStage",
                      modes=[PipelineModes.OTHER]))
