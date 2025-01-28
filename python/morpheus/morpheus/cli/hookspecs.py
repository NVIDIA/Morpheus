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

import typing

import click
import pluggy

from morpheus.cli.stage_registry import StageRegistry
from morpheus.config import PipelineModes

hookspec = pluggy.HookspecMarker("morpheus")

# pylint:disable=unused-argument


@hookspec
def morpheus_cli_collect_stages(registry: StageRegistry):
    pass


@hookspec
def morpheus_cli_collect_stage_names(mode: PipelineModes) -> typing.List[str]:
    pass


@hookspec(firstresult=True)
def morpheus_cli_make_stage_command(mode: PipelineModes, stage_name: str) -> click.Command:
    pass
