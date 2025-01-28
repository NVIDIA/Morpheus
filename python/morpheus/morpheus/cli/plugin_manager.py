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

import collections
import importlib
import os
import sys
import types
import typing

import pluggy

from morpheus.cli import hookspecs
from morpheus.cli.default_command_hooks import DefaultCommandHooks
from morpheus.cli.stage_registry import GlobalStageRegistry
from morpheus.cli.stage_registry import StageRegistry
from morpheus.cli.utils import PluginSpec


class PluginManager():

    _singleton: "PluginManager" = None

    def __init__(self):
        self._pm = pluggy.PluginManager("morpheus")
        self._pm.add_hookspecs(hookspecs)
        self._pm.register(DefaultCommandHooks(), name="morpheus_default")

        self._plugins_loaded = False
        self._plugin_specs: typing.List[PluginSpec] = []

        self._stage_registry: StageRegistry = None

    def _get_plugin_specs_as_list(self, specs: PluginSpec) -> typing.List[str]:
        """Parse a plugins specification into a list of plugin names."""
        # None means empty.
        if specs is None:
            return []
        # Workaround for #3899 - a submodule which happens to be called "pytest_plugins".
        if isinstance(specs, types.ModuleType):
            return []
        # Comma-separated list.
        if isinstance(specs, str):
            return specs.split(",") if specs else []
        # Direct specification.
        if isinstance(specs, collections.abc.Sequence):
            return list(specs)
        raise RuntimeError(
            f"Plugins may be specified as a sequence or a ','-separated string of plugin names. Got: {specs}")

    def _ensure_plugins_loaded(self):

        if (self._plugins_loaded):
            return

        # # Now that all command line plugins have been added, add any from the env variable
        self.add_plugin_option(os.environ.get("MORPHEUS_PLUGINS"))

        # Loop over all specs and load the plugins
        for spec in self._plugin_specs:
            try:
                if os.path.exists(spec):
                    mod_name = os.path.splitext(os.path.basename(spec))[0]
                    spec = importlib.util.spec_from_file_location(mod_name, spec)
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[mod_name] = mod
                    spec.loader.exec_module(mod)
                else:
                    mod = importlib.import_module(spec)

                # Sucessfully loaded. Register
                self._pm.register(mod)

            except ImportError as e:
                raise ImportError(f'Error importing plugin "{spec}": {e.args[0]}').with_traceback(
                    e.__traceback__) from e

        # Finally, consider setuptools entrypoints
        self._pm.load_setuptools_entrypoints("morpheus")

        self._plugins_loaded = True

    def add_plugin_option(self, spec: PluginSpec):
        # Append to the list of specs
        self._plugin_specs.extend(self._get_plugin_specs_as_list(spec))

    def get_registered_stages(self) -> StageRegistry:

        if (self._stage_registry is None):

            self._ensure_plugins_loaded()

            # Start with the global registry (optionally make a clone?)
            self._stage_registry = GlobalStageRegistry.get()

            # Now call the plugin system to add stages as necessary
            # pylint: disable=no-member
            self._pm.hook.morpheus_cli_collect_stages(registry=self._stage_registry)

        return self._stage_registry

    @staticmethod
    def get() -> "PluginManager":
        if (PluginManager._singleton is None):
            PluginManager._singleton = PluginManager()
        return PluginManager._singleton
