import dataclasses
import time
import typing

import click

from morpheus.config import PipelineModes


@dataclasses.dataclass
class StageInfo:
    name: str
    modes: typing.List[PipelineModes]
    build_command: typing.Callable[[], click.Command]

    def __post_init__(self):
        # If modes is None or empty, then convert it to all modes
        if (self.modes is None or len(self.modes) == 0):
            self.modes = [x for x in PipelineModes]

    def supports_mode(self, mode: PipelineModes):
        if (mode is None):
            return True

        if (self.modes is None or len(self.modes) == 0):
            return True

        return mode in self.modes


@dataclasses.dataclass
class LazyStageInfo(StageInfo):

    qualified_name: str
    package_name: str
    class_name: str

    def __init__(self, name: str, stage_qualified_name: str, modes: typing.List[PipelineModes]):

        super().__init__(name, modes, self._lazy_build)

        self.qualified_name = stage_qualified_name

        # Break the module name up into the class and the package
        qual_name_split = stage_qualified_name.split(".")
        if (len(qual_name_split) > 1):
            self.package_name = ".".join(qual_name_split[:-1])

        self.class_name = qual_name_split[-1]

    def _lazy_build(self):

        start_time = time.time()

        import importlib

        mod = importlib.import_module(self.package_name)

        # Now get the class name
        stage_class = getattr(mod, self.class_name, None)

        if (stage_class is None):
            raise RuntimeError("Could not import {} from {}".format(self.class_name, self.package_name))

        # Now get the stage info from the class (it must have been registered during the import)
        stage_class_info: StageInfo = getattr(stage_class, "_morpheus_registered_stage", None)

        if (stage_class_info is None):
            raise RuntimeError(
                "Class {} did not have attribute '_morpheus_registered_stage'. Did you use register_stage?".format(
                    self.qualified_name))

        print("Loading load of '{}' took: {} ms".format(self.name, (time.time() - start_time) * 1000.0))

        return stage_class_info.build_command()


class StageRegistry:

    def __init__(self) -> None:

        # Stages are registered on a per mode basis so different stages can have the same command name for different modes
        self._registered_stages: typing.Dict[PipelineModes, typing.Dict[str, StageInfo]] = {}

    def _get_stages_for_mode(self, mode: PipelineModes) -> typing.Dict[str, StageInfo]:

        if (mode not in self._registered_stages):
            self._registered_stages[mode] = {}

        return self._registered_stages[mode]

    def _add_stage_info(self, mode: PipelineModes, stage: StageInfo):

        # Get the stages for the mode
        mode_stages = self._get_stages_for_mode(mode)

        if (stage.name in mode_stages):
            raise RuntimeError("The stage '{}' has already been added for mode: {}".format(stage.name, mode))

        mode_stages[stage.name] = stage

    def add_stage_info(self, stage: StageInfo):

        # Loop over all modes for the stage
        for m in stage.modes:
            self._add_stage_info(m, stage)

    def get_stage_info(self, stage_name: str, mode: PipelineModes = None, raise_missing=False) -> StageInfo:

        mode_registered_stags = self._get_stages_for_mode(mode)

        if (stage_name not in mode_registered_stags):
            if (raise_missing):
                raise RuntimeError("Could not find stage '{}' in registry".format(stage_name))
            else:
                return None

        stage_info = mode_registered_stags[stage_name]

        # Now check the modes
        if (stage_info.supports_mode(mode)):
            return stage_info

        # Found but no match on mode
        if (raise_missing):
            raise RuntimeError("Found stage '{}' in registry, but it does not support pipeline mode: {}".format(
                stage_name, mode))
        else:
            return None

    def get_registered_names(self, mode: PipelineModes = None) -> typing.List[str]:

        # Loop over all registered stages and validate the mode
        stage_names: typing.List[str] = [
            name for name, stage_info in self._get_stages_for_mode(mode).items() if stage_info.supports_mode(mode)
        ]

        return stage_names


class GlobalStageRegistry:

    _global_registry: StageRegistry = StageRegistry()

    @staticmethod
    def get() -> StageRegistry:
        return GlobalStageRegistry._global_registry
