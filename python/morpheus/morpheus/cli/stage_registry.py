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

import dataclasses
import logging
import typing

import click

from morpheus.config import PipelineModes

logger = logging.getLogger(__file__)


@dataclasses.dataclass
class StageInfo:
    name: str  # The command Name
    modes: typing.List[PipelineModes]
    qualified_name: str  # The fully qualified name of the stage. Only used for comparison
    build_command: typing.Callable[[], click.Command] = dataclasses.field(compare=False, repr=False)

    def __post_init__(self):
        # If modes is None or empty, then convert it to all modes
        if (self.modes is None or len(self.modes) == 0):
            self.modes = set(PipelineModes)

    def supports_mode(self, mode: PipelineModes):
        if (mode is None):
            return True

        if (self.modes is None or len(self.modes) == 0):
            return True

        return mode in self.modes


@dataclasses.dataclass
class LazyStageInfo(StageInfo):

    package_name: str
    class_name: str

    def __init__(self, name: str, stage_qualified_name: str, modes: typing.List[PipelineModes]):

        super().__init__(name=name, modes=modes, qualified_name=stage_qualified_name, build_command=self._lazy_build)

        # Break the module name up into the class and the package
        qual_name_split = stage_qualified_name.split(".")
        if (len(qual_name_split) > 1):
            self.package_name = ".".join(qual_name_split[:-1])

        self.class_name = qual_name_split[-1]

    def _lazy_build(self):

        import importlib

        mod = importlib.import_module(self.package_name)

        # Now get the class name
        stage_class = getattr(mod, self.class_name, None)

        if (stage_class is None):
            raise RuntimeError(f"Could not import {self.class_name} from {self.package_name}")

        # Now get the stage info from the class (it must have been registered during the import)
        stage_class_info: StageInfo = getattr(stage_class, "_morpheus_registered_stage", None)

        if (stage_class_info is None):
            raise RuntimeError(f"Class {self.qualified_name} did not have attribute '_morpheus_registered_stage'. \
                Did you use register_stage?")

        return stage_class_info.build_command()


class StageRegistry:

    def __init__(self) -> None:
        # Stages are registered on a per mode basis, different stages can have the same command name for different modes
        self._registered_stages: typing.Dict[PipelineModes, typing.Dict[str, StageInfo]] = {}

    def _get_stages_for_mode(self, mode: PipelineModes) -> typing.Dict[str, StageInfo]:

        if (mode not in self._registered_stages):
            self._registered_stages[mode] = {}

        return self._registered_stages[mode]

    def _add_stage_info(self, mode: PipelineModes, stage: StageInfo):

        # Get the stages for the mode
        mode_stages = self._get_stages_for_mode(mode)

        if (stage.name in mode_stages):
            # TODO(MDD): Figure out if this is something that only the unittests encounter
            logging.debug("The stage %s has already been added for mode: %s", stage.name, mode)

        mode_stages[stage.name] = stage

    def add_stage_info(self, stage: StageInfo):

        # Loop over all modes for the stage
        for mode in stage.modes:
            self._add_stage_info(mode, stage)

    def get_stage_info(self, stage_name: str, mode: PipelineModes = None, raise_missing=False) -> StageInfo:

        mode_registered_stags = self._get_stages_for_mode(mode)

        if (stage_name not in mode_registered_stags):
            if (raise_missing):
                raise RuntimeError(f"Could not find stage '{stage_name}' in registry")

            return None

        stage_info = mode_registered_stags[stage_name]

        # Now check the modes
        if (stage_info.supports_mode(mode)):
            return stage_info

        # Found but no match on mode
        if (raise_missing):
            raise RuntimeError("Found stage '{stage_name}' in registry, but it does not support pipeline mode: {mode}")

        return None

    def get_registered_names(self, mode: PipelineModes = None) -> typing.List[str]:

        # Loop over all registered stages and validate the mode
        stage_names: typing.List[str] = [
            name for name, stage_info in self._get_stages_for_mode(mode).items() if stage_info.supports_mode(mode)
        ]

        return stage_names

    def _remove_stage_info(self, mode: PipelineModes, stage: StageInfo):

        # Get the stages for the mode
        mode_stages = self._get_stages_for_mode(mode)

        if (stage.name not in mode_stages):
            # TODO(MDD): Figure out if this is something that only the unittests encounter
            logging.debug("The stage '%s' has already been added for mode: %s", stage.name, mode)

        mode_stages.pop(stage.name)

    def remove_stage_info(self, stage: StageInfo):

        # Loop over all modes for the stage
        for mode in stage.modes:
            self._remove_stage_info(mode, stage)


class GlobalStageRegistry:

    _global_registry: StageRegistry = StageRegistry()

    @staticmethod
    def get() -> StageRegistry:
        return GlobalStageRegistry._global_registry
