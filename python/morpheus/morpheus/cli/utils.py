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
"""Misc. utilities for the CLI"""

import logging
import os
import types
import typing
import warnings
from enum import Enum
from functools import update_wrapper

import click
import click.globals

import morpheus
from morpheus.config import Config
from morpheus.config import ConfigBase
# For backwards compatibility
from morpheus.utils.file_utils import load_labels_file  # pylint: disable=unused-import # noqa: F401

# Ignore pipeline unless we are typechecking since it takes a while to import
if (typing.TYPE_CHECKING):
    from morpheus.pipeline.linear_pipeline import LinearPipeline

logger = logging.getLogger(__name__)

PluginSpec = typing.Union[None, types.ModuleType, str, typing.Sequence[str]]


def str_to_file_type(file_type_str: str):
    """Converts a string to a FileType enum member"""
    # Delay FileTypes since this will import ._lib
    from morpheus.common import FileTypes
    file_type_members = {name.lower(): t for (name, t) in FileTypes.__members__.items()}

    return file_type_members[file_type_str]


def _without_empty_args(passed_args):
    return {k: v for k, v in passed_args.items() if v is not None}


def is_pybind_enum(cls: typing.Any):
    """
    Determines if the given `cls` is an enum.
    C++ enums exposed via pybind11 do not inherit from Enum, but do expose the `__members__` convention.
    https://docs.python.org/3.10/library/enum.html?highlight=__members__#iteration
    """
    return 'pybind11' in type(cls).__name__ and hasattr(cls, '__members__')


def without_empty_args(f):
    """Removes keyword arguments that have a None value"""

    def new_func(*args, **kwargs):
        kwargs = _without_empty_args(kwargs)
        return f(click.globals.get_current_context(), *args, **kwargs)

    return update_wrapper(new_func, f)


def show_defaults(f):
    """Ensures the click.Context has `show_defaults` set to True. (Seems like a bug currently)"""

    def new_func(*args, **kwargs):
        ctx: click.Context = click.globals.get_current_context()
        ctx.show_default = True
        return f(*args, **kwargs)

    return update_wrapper(new_func, f)


def _apply_to_config(config: ConfigBase, **kwargs):
    for (param, value) in kwargs.items():
        if hasattr(config, param):
            setattr(config, param, value)
        else:
            warnings.warn(f"No config option matches for {param}")

    return config


def prepare_command(parse_config: bool = False):
    """Decorator ensuring that a click.Context object is passed as the first argument to the decorated function"""

    def inner_prepare_command(f):
        """Preparse command for use. Combines @without_empty_args, @show_defaults and @click.pass_context"""

        def new_func(*args, **kwargs):
            ctx: click.Context = click.globals.get_current_context()
            ctx.show_default = True

            # Set the max width. This will still default to the users console width
            ctx.max_content_width = 200

            kwargs = _without_empty_args(kwargs)

            # Apply the config if desired
            if parse_config:
                config = get_config_from_ctx(ctx)

                _apply_to_config(config, **kwargs)

            return f(ctx, *args, **kwargs)

        return update_wrapper(new_func, f)

    return inner_prepare_command


def get_config_from_ctx(ctx: click.Context) -> Config:
    """Returns the config from a click context"""
    ctx_dict = ctx.ensure_object(dict)

    if "config" not in ctx_dict:
        ctx_dict["config"] = Config()

    return ctx_dict["config"]


def get_pipeline_from_ctx(ctx: click.Context) -> "LinearPipeline":
    """Returns the pipeline from a click context"""
    ctx_dict = ctx.ensure_object(dict)

    assert "pipeline" in ctx_dict, "Inconsistent configuration. Pipeline accessed before created"

    return ctx_dict["pipeline"]


morpheus_log_levels = list(logging._nameToLevel.keys())

if ("NOTSET" in morpheus_log_levels):
    morpheus_log_levels.remove("NOTSET")


def get_log_levels():
    """Returns a list of all available logging levels in string format"""
    return morpheus_log_levels


# pylint: disable=unused-argument
def parse_log_level(ctx, param, value):
    """Click callback that parses a command line value into a logging level"""
    x = logging._nameToLevel.get(value.upper(), None)
    if x is None:
        raise click.BadParameter(f'Must be one of {", ".join(logging._nameToLevel.keys())}. Passed: {value}')
    return x


def is_enum(enum_class: typing.Type):
    """Returns True if the given class is an enum."""
    try:
        return issubclass(enum_class, Enum) or is_pybind_enum(enum_class)
    except TypeError:
        return False


def get_enum_members(enum_class: typing.Type):
    """Returns a dict of enum members for the given enum."""
    assert is_enum(enum_class), "Must pass a class that derives from Enum or a C++ enum exposed to Python"

    return dict(enum_class.__members__)


def get_enum_keys(enum_class: typing.Type):
    """Returns a list of keys for the given enum."""
    enum_map = get_enum_members(enum_class)

    return list(enum_map.keys())


def parse_enum(_: click.Context, _2: click.Parameter, value: str, enum_class: typing.Type, case_sensitive=True):
    """Parses a string into an enum value."""
    enum_map = get_enum_members(enum_class)

    if not case_sensitive:
        # Make the keys lowercase
        enum_map = {key.lower(): value for key, value in enum_map.items()}
        value = value.lower()

    result = enum_map[value]

    return result


def get_package_relative_file(filename: str):
    """
    If `filename` is a relative path, and does not exist, attempt to locate the file relative to the directory of the
    `morpheus` package, if it exists return the abolute path of the file found. Otherwise returns `filename` unchanged.
    """
    # First check if the path is relative
    if (not os.path.isabs(filename)):

        # See if the file exists.
        does_exist = os.path.exists(filename)

        if (not does_exist):
            # If it doesn't exist, then try to make it relative to the morpheus library root
            morpheus_root = os.path.dirname(morpheus.__file__)

            value_abs_to_root = os.path.join(morpheus_root, filename)

            # If the file relative to our package exists, use that instead
            if (os.path.exists(value_abs_to_root)):

                return value_abs_to_root

    return filename


class MorpheusRelativePath(click.Path):
    """
    A specialization of the `click.Path` class that falls back to using package relative paths if the file cannot be
    found. Takes the exact same parameters as `click.Path`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Append "data" to the name so it can be different than normal click.Path
        self.name = "data " + self.name

    def convert(self,
                value: typing.Any,
                param: typing.Optional["click.Parameter"],
                ctx: typing.Optional["click.Context"]) -> typing.Any:
        """Wrapper around `get_package_relative_file` for `click.Path`."""
        package_relative = get_package_relative_file(value)

        if (package_relative != value):
            logger.debug(("Parameter, '%s', with relative path, '%s', does not exist. "
                          "Using package relative location: '%s'"),
                         param.name,
                         value,
                         package_relative)

        return super().convert(package_relative, param, ctx)
