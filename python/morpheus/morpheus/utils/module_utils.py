# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
"""Module utilities for Morpheus."""

import functools
import logging
import re
import typing
from typing import Optional
from typing import Type

import mrc
from pydantic import BaseModel

from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_utils import get_df_pkg_from_obj

logger = logging.getLogger(__name__)

Registry = mrc.ModuleRegistry
mrc_version = [int(i) for i in mrc.__version__.split('.')]


def verify_module_registration(func):
    """
    Module availability in the module registry is verified by this function.

    Parameters
    ----------
    func : Function that requires wrapping.

    Returns
    -------
    inner_func
        Encapsulated function.
    """

    @functools.wraps(func)
    def inner_func(config, **kwargs):

        verify_module_meta_fields(config)

        module_id = config.get("module_id")
        namespace = config.get("namespace")

        if module_id is None or namespace is None:
            raise TypeError("TypeError: a string-like object is required for module_id and namespace, not 'NoneType'")

        if not Registry.contains(module_id, namespace):
            raise ValueError(f"Module '{module_id}' doesn't exist in the namespace '{namespace}'")

        return func(config, **kwargs)

    return inner_func


def register_module(module_id, namespace):
    """
    Registers a module if not exists in the module registry.

    Parameters
    ----------
    module_id : str
        Unique identifier for a module in the module registry.
    namespace : str
        Namespace to virtually cluster the modules.

    Returns
    -------
    inner_func
        Encapsulated function.
    """

    def inner_func(func):
        # Register a module if not exists in the registry.
        if not Registry.contains(module_id, namespace):
            Registry.register_module(module_id, namespace, mrc_version, func)
            logger.debug("Module '%s' was successfully registered with '%s' namespace.", module_id, namespace)
        else:
            logger.debug("Module: '%s' already exists in the given namespace '%s'", module_id, namespace)

        return func

    return inner_func


@verify_module_registration
def load_module(config: typing.Dict, builder: mrc.Builder):
    """
    Loads a module that exists in the module registry.

    Parameters
    ----------
    config : typing.Dict
        Module configuration.
    builder : mrc.Builder
        MRC Builder object.

    Returns
    -------
    module
        Module object.
    """

    module_id = config.get("module_id")
    namespace = config.get("namespace")
    module_name = config.get("module_name")

    module = builder.load_module(module_id, namespace, module_name, config)

    logger.debug("Module '%s' with namespace '%s' is successfully loaded.", module_id, namespace)

    return module


def verify_module_meta_fields(config: typing.Dict):
    """
    This function make sure the module configuration contains meta fields.

    Parameters
    ----------
    config : typing.Dict
        Module configuration.
    """

    if "module_id" not in config:
        raise KeyError("Required attribute 'module_id' is missing in the module configuration.")
    if "namespace" not in config:
        raise KeyError("Required attribute 'namespace' is missing in the module configuration.")
    if "module_name" not in config:
        raise KeyError("Required attribute 'module_name' is missing in the module configuration.")


def merge_dictionaries(primary_dict, secondary_dict):
    """Recursively merge two dictionaries, using primary_dict as a tie-breaker.

    Lists are treated as a special case, and all unique elements from both dictionaries are included in the final list.

    Args:
        primary_dict (dict): The primary dictionary.
        secondary_dict (dict): The secondary dictionary.

    Returns:
        dict: The merged dictionary.
    """
    result_dict = primary_dict.copy()

    for key, value in secondary_dict.items():
        if key in result_dict:
            if isinstance(value, list) and isinstance(result_dict[key], list):
                # Combine the two lists and remove duplicates while preserving order
                # This isn't perfect, its possible we could still end up with duplicates in some scenarios
                combined_list = result_dict[key] + value
                unique_list = []
                for item in combined_list:
                    if item not in unique_list:
                        unique_list.append(item)
                result_dict[key] = unique_list
            elif isinstance(value, dict) and isinstance(result_dict[key], dict):
                # Recursively merge the two dictionaries
                result_dict[key] = merge_dictionaries(result_dict[key], value)
        else:
            result_dict[key] = value

    return result_dict


period_to_strptime = {
    "s": "%Y-%m-%d %H:%M:%S",
    "T": "%Y-%m-%d %H:%M",
    "min": "%Y-%m-%d %H:%M",
    "H": "%Y-%m-%d %H",
    "D": "%Y-%m-%d",
    "W": "%Y-%U",
    "M": "%Y-%m",
    "Y": "%Y"
}


def to_period_approximation(data_df: DataFrameType, period: str) -> DataFrameType:
    """
    This function converts a dataframe to a period approximation.

    Parameters
    ----------
    data_df : DataFrameType
        Input cudf/pandas dataframe.
    period : str
        Period.

    Returns
    -------
    DataFrame
        Period approximation of the input cudf/pandas dataframe.
    """

    match = re.match(r"(\d*)(\w)", period)
    if not match:
        raise ValueError(f"Invalid period format: {period}.")

    if period not in period_to_strptime:
        raise ValueError(f"Unknown period: {period}. Supported period: {period_to_strptime}")

    strptime_format = period_to_strptime[period]

    df_pkg = get_df_pkg_from_obj(data_df)
    data_df["period"] = df_pkg.to_datetime(data_df["ts"].dt.strftime(strptime_format) + '-1',
                                           format=f"{strptime_format}-%w")

    return data_df


def get_config_with_overrides(config, module_id, module_name=None, module_namespace="morpheus"):
    """This function returns the module configuration with the overrides."""
    sub_config = config.get(module_id, None)

    try:
        if module_name is None:
            module_name = sub_config.get("module_name")
    except Exception as exc_info:
        raise KeyError(f"'module_name' is not set in the '{module_id}' module configuration.") from exc_info

    sub_config.setdefault("module_id", module_id)
    sub_config.setdefault("module_name", module_name)
    sub_config.setdefault("namespace", module_namespace)

    return sub_config


def get_module_config(module_id: str, builder: mrc.Builder):
    """
    Returns the module configuration for the specified module id.

    Parameters
    ----------
    module_id : str
        Unique identifier for a module in the module registry.
    builder : mrc.Builder
        MRC Builder object.

    Returns
    -------
    config : typing.Dict
        Module configuration.

    """

    config = builder.get_current_module_config()
    if module_id in config:
        config = config[module_id]

    verify_module_meta_fields(config)

    return config


def make_nested_module(module_id: str, namespace: str, ordered_modules_meta: typing.List[typing.Dict[str, str]]):
    """
    This function creates a nested module and registers it in the module registry.
    This module unifies a chain of two or more modules into a single module.

    Parameters
    ----------
    module_id : str
        Unique identifier for a module in the module registry.
    namespace : str
        Namespace to virtually cluster the modules.
    ordered_modules_meta : typing.List[typing.Dict[str, str]]
        The sequence in which the edges between the nodes are made will be determined by ordered modules meta.
    """

    @register_module(module_id, namespace)
    def module_init(builder: mrc.Builder):

        prev_module = None
        head_module = None

        # Make edges between set of modules and wrap the internally connected modules as module.
        #                        Wrapped Module
        #                _______________________________
        #
        #    input >>   | Module1 -- Module2 -- Module3 |   >> output
        #                ________________________ ______
        for config in ordered_modules_meta:

            curr_module = load_module(config, builder=builder)

            if prev_module:
                builder.make_edge(prev_module.output_port("output"), curr_module.input_port("input"))
            else:
                head_module = curr_module

            prev_module = curr_module

        # Register input and output port for a module.
        builder.register_module_input("input", head_module.input_port("input"))
        builder.register_module_output("output", prev_module.output_port("output"))


class ModuleLoader:
    """
    Class to hold the definition of a module.

    Attributes
    ----------
    module_instance : ModuleLoader
        The instance of the loaded module.
    name : str
        The name of the module.
    config : dict
        The configuration dictionary for the module.
    """

    def __init__(self, module_interface, name, config):
        self._module_interface = module_interface
        self._name = name
        self._config = config
        self._loaded = False

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

    def load(self, builder: mrc.Builder):
        """
        Loads the module instance.

        Parameters
        ----------
        builder : mrc.Builder
            The Morpheus builder instance.
        """

        if (self._loaded):
            err_msg = f"Module '{self._module_interface.identity}::{self.name}' is already loaded."
            logger.error(err_msg)

            raise RuntimeError(err_msg)

        module = builder.load_module(self._module_interface.identity,
                                     self._module_interface.namespace,
                                     self._name,
                                     self._config)

        logger.debug("Module '%s' with namespace '%s' is successfully loaded.",
                     self._module_interface.identity,
                     self._module_interface.namespace)

        self._loaded = True

        return module


class ModuleLoaderFactory:
    """
    Class that acts as a simple wrapper to load a SegmentModule.

    Attributes
    ----------
    _id : str
        The module identifier.
    _namespace : str
        The namespace of the module.
    _config_schema : Type[BaseModel], optional
        The Pydantic model representing the parameter contract for the module.
    """

    def __init__(self, module_id, module_namespace, config_schema: Optional[Type[BaseModel]] = None):
        self._id = module_id
        self._namespace = module_namespace
        self._config_schema = config_schema

    @property
    def identity(self):
        return self._id

    @property
    def namespace(self):
        return self._namespace

    def get_instance(self, module_name: str, module_config: dict) -> ModuleLoader:
        """
        Loads a module instance and returns its definition.

        Parameters
        ----------
        module_name : str
            The name of the module to be loaded.
        module_config : dict
            The configuration dictionary for the module.

        Returns
        -------
        ModuleLoader
            A specific instance of this module.
        """
        return ModuleLoader(self, module_name, module_config)

    def print_schema(self) -> str:
        """
        Returns a human-readable description of the module's parameter schema.

        Returns
        -------
        str
            A description of the module's parameter schema.
        """
        if not self._config_schema:
            return "No parameter contract defined for this module."

        description = f"Schema for {self._id}:\n"
        for field in self._config_schema.__fields__.values():
            description += f"  - {field.name} ({field.type_.__name__}): {field.field_info.description}\n"

        return description
