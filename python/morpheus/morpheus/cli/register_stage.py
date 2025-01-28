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

import functools
import inspect
import pathlib
import re
import typing

import click
import numpydoc.docscrape
from typing_utils import get_args
from typing_utils import issubtype

from morpheus.cli.stage_registry import GlobalStageRegistry
from morpheus.cli.stage_registry import LazyStageInfo
from morpheus.cli.stage_registry import StageInfo
from morpheus.cli.utils import get_config_from_ctx
from morpheus.cli.utils import get_enum_keys
from morpheus.cli.utils import get_pipeline_from_ctx
from morpheus.cli.utils import is_enum
from morpheus.cli.utils import parse_enum
from morpheus.cli.utils import prepare_command
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.utils.type_utils import _DecoratorType
from morpheus.utils.type_utils import get_full_qualname
from morpheus.utils.type_utils import is_union_type


def class_name_to_command_name(class_name: str) -> str:
    # Convert to snake case with dashes
    snake_case = re.sub(r'(?<!^)(?=[A-Z])', '-', class_name).lower()

    # remove trailing "stage"
    return re.sub(r'-stage$', '', snake_case)


def get_param_doc(numpydoc_obj: numpydoc.docscrape.NumpyDocString, name: str):
    found_doc = next((x for x in numpydoc_obj["Parameters"] if x.name == name), None)

    if (not found_doc):
        return ""

    # Combine the docstrings into a single string
    param_doc = " ".join(found_doc.desc)

    return param_doc


def get_param_type(numpydoc_obj: numpydoc.docscrape.NumpyDocString, name: str):
    found_doc = next((x for x in numpydoc_obj["Parameters"] if x.name == name), None)

    if (not found_doc):
        return ""

    return found_doc.type


def parse_type_value(value_str: str) -> typing.Any:  # pylint: disable=too-many-return-statements

    value_lower = value_str.lower()

    # Check for bool
    if (value_lower == "true"):
        return True
    if (value_lower == "false"):
        return False
    if (value_lower == "none"):
        return None
    if (value_str.startswith('"') and value_str.endswith('"')):
        return value_str.strip('"')
    if (value_str.startswith("'") and value_str.endswith("'")):
        return value_str.strip("'")

    # Try to parse as float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Try to parse as int
    try:
        return int(value_str)
    except ValueError:
        pass

    # Otherwise just return none
    return None


def parse_doc_type_str(doc_type_str: str) -> dict:

    # Split along ,
    comma_split = doc_type_str.split(",")

    out_dict = {}

    # Now loop over comma separated
    for x in comma_split:
        # Split on =
        equal_split = x.split("=")

        if (len(equal_split) == 2):
            # Key value pair
            out_dict[equal_split[0].strip()] = parse_type_value(equal_split[1].strip())
        elif (len(equal_split) == 1):
            # Single type
            out_dict[""] = equal_split[0].strip()
        else:
            raise RuntimeError(f"Invalid docstring: {doc_type_str}")

    return out_dict


def get_doc_kwargs(doc_type_str: str) -> dict:

    out_dict = parse_doc_type_str(doc_type_str)

    # pop out the empty and default
    out_dict.pop("", None)
    out_dict.pop("default", None)

    return out_dict


def partial_pop_kwargs(function, input_dict: dict):
    """
    Binds any matching kwargs in `input_dict` to arguments in `function`. Any that match are popped from the dictionary.
    Returned value is a `functools.partial` with matching arguments bound
    """

    # Get the function signature
    fn_sig = inspect.signature(function)

    bound_args = {
        input_name: input_dict.pop(input_name)
        for input_name in list(input_dict.keys()) if input_name in fn_sig.parameters
    }

    return functools.partial(function, **bound_args)


def has_matching_kwargs(function, input_dict: dict) -> bool:
    # Check for any matching arguments between function and input_dict

    # Get the function signature
    fn_sig = inspect.signature(function)

    return len([True for input_name in list(input_dict.keys()) if input_name in fn_sig.parameters]) > 0


def _convert_enum_default(options_kwargs: dict, annotation):
    """
    Display the default value of an enum argument as a string not an enum instance
    """
    default_val = options_kwargs.get('default')
    if (isinstance(default_val, annotation)):
        options_kwargs['default'] = default_val.name


def set_options_param_type(options_kwargs: dict, annotation, doc_type: str):

    doc_type_kwargs = get_doc_kwargs(doc_type)

    if (annotation == inspect.Parameter.empty):
        raise RuntimeError("All types must be specified to auto register stage.")

    if (is_union_type(annotation)):
        raise RuntimeError("Union types are not supported for auto registering stages.")

    if (issubtype(annotation, pathlib.Path)):
        # For paths, use the Path option and apply any kwargs
        options_kwargs["type"] = partial_pop_kwargs(click.Path, doc_type_kwargs)()

    elif (is_enum(annotation)):
        case_sensitive = doc_type_kwargs.get('case_sensitive', True)
        options_kwargs["type"] = partial_pop_kwargs(click.Choice, doc_type_kwargs)(get_enum_keys(annotation))

        _convert_enum_default(options_kwargs, annotation)
        options_kwargs["callback"] = functools.partial(parse_enum, enum_class=annotation, case_sensitive=case_sensitive)

    elif (issubtype(annotation, int) and not issubtype(annotation, bool)):
        # Check if there are any range arguments. Otherwise use a normal int
        if (has_matching_kwargs(click.IntRange, doc_type_kwargs)):
            options_kwargs["type"] = partial_pop_kwargs(click.IntRange, doc_type_kwargs)()
        else:
            options_kwargs["type"] = annotation

    elif (issubtype(annotation, float)):
        # Check if there are any range arguments. Otherwise use a normal int
        if (has_matching_kwargs(click.FloatRange, doc_type_kwargs)):
            options_kwargs["type"] = partial_pop_kwargs(click.FloatRange, doc_type_kwargs)()
        else:
            options_kwargs["type"] = annotation

    elif (issubtype(annotation, dict)):
        options_kwargs["multiple"] = True
        options_kwargs["type"] = click.Tuple([str, str])
        options_kwargs["callback"] = lambda ctx, param, value: dict(value)

    elif (issubtype(annotation, typing.List) or issubtype(annotation, typing.Tuple)
          or issubtype(annotation, typing.Iterable)) and not (issubtype(annotation, str)):

        # For variable length array, use multiple=True
        options_kwargs["multiple"] = True
        options_kwargs["type"] = get_args(annotation)[0]

    else:
        options_kwargs["type"] = annotation

    # Update any remaining docstring kwargs
    options_kwargs.update(doc_type_kwargs)


def compute_option_name(stage_arg_name: str, rename_options: typing.Dict[str, str] = None) -> tuple:

    if (rename_options is None):
        rename_options = {}

    rename_val = rename_options.get(stage_arg_name, f"--{stage_arg_name}")

    if (issubtype(type(rename_val), str)):
        rename_val = (rename_val, )
    elif (not issubtype(type(rename_val), tuple)):
        rename_val = tuple(rename_val)

    for name in rename_val:
        if (not name.startswith("-")):
            raise RuntimeError(
                f"Rename value '{name}' for option '{stage_arg_name}', must start with '-'. i.e. '--my_new_option")

    # Create the click option name as a ("stage_arg_name", "--rename1", "--rename2", "-r")
    return (stage_arg_name, ) + rename_val


def register_stage(command_name: str = None,
                   modes: typing.Sequence[PipelineModes] = None,
                   ignore_args: typing.List[str] = None,
                   command_args: dict = None,
                   option_args: typing.Dict[str, dict] = None,
                   rename_options: typing.Dict[str, str] = None):

    if (ignore_args is None):
        ignore_args = []

    if (command_args is None):
        command_args = {}

    if (option_args is None):
        option_args = {}

    if (rename_options is None):
        rename_options = {}

    if (modes is None):
        modes = list(PipelineModes)

    def register_stage_inner(stage_class: _DecoratorType) -> _DecoratorType:

        nonlocal command_name

        # A subclass of a stage that is already registered with the CLI will already have this attribute set,
        # but the command name won't match.
        if (not hasattr(stage_class, "_morpheus_registered_stage")
                or stage_class._morpheus_registered_stage.name != command_name):

            # Determine the command name if it wasnt supplied
            if (command_name is None):
                command_name = class_name_to_command_name(stage_class.__name__)

            def build_command():

                command_params: typing.List[click.Parameter] = []

                class_init_sig = inspect.signature(stage_class)

                config_param_name = None

                numpy_doc: numpydoc.docscrape.NumpyDocString = None

                # Try to get the docstring from __init__first, then from the class itself
                if (hasattr(stage_class, "__init__") and stage_class.__init__.__doc__ is not None):
                    numpy_doc = numpydoc.docscrape.get_doc_object(stage_class.__init__)
                else:
                    numpy_doc = numpydoc.docscrape.get_doc_object(stage_class)

                for p_name, p_value in class_init_sig.parameters.items():

                    try:
                        if (p_value.annotation == Config):
                            config_param_name = p_name
                            continue

                        if (p_name in ignore_args):
                            assert p_value.default != inspect.Parameter.empty, (
                                "Cannot ignore argument without default value")
                            continue

                        if (p_value.kind == inspect.Parameter.VAR_POSITIONAL):
                            continue

                        if (p_value.kind == inspect.Parameter.VAR_KEYWORD):
                            continue

                        option_kwargs = {}

                        # See if we have some sort of documentation for this argument
                        option_kwargs["help"] = get_param_doc(numpy_doc, p_name)

                        # Set the default value if not empty
                        if p_value.default != inspect.Parameter.empty:
                            option_kwargs["required"] = False
                            option_kwargs["default"] = p_value.default
                        else:
                            option_kwargs["required"] = True
                            option_kwargs["default"] = None

                        set_options_param_type(option_kwargs, p_value.annotation, get_param_type(numpy_doc, p_name))

                        # Now overwrite with any user supplied values for this option
                        option_kwargs.update(option_args.get(p_name, {}))

                        # Get the name settings
                        click_option_name = compute_option_name(p_name, rename_options)

                        option = click.Option(click_option_name, **option_kwargs)

                        command_params.append(option)
                    except Exception as ex:
                        raise RuntimeError((f"Error auto registering CLI command '{command_name}' with "
                                            f"class '{stage_class}' and parameter '{p_name}'. Error: {ex}")) from ex

                if (config_param_name is None):
                    raise RuntimeError("All stages must take on argument of morpheus.Config. Ensure your stage "
                                       "constructor as one argument that has been annotated with morpheus.Config. "
                                       "i.e. `c: morpheus.Config`")

                def command_callback(ctx: click.Context, **kwargs):

                    # Delay loading SourceStage
                    from morpheus.pipeline.source_stage import SourceStage

                    config = get_config_from_ctx(ctx)
                    pipeline = get_pipeline_from_ctx(ctx)

                    # Set the config to the correct parameter
                    kwargs[config_param_name] = config

                    stage = stage_class(**kwargs)

                    if (issubclass(stage_class, SourceStage)):
                        pipeline.set_source(stage)
                    else:
                        pipeline.add_stage(stage)

                    return stage

                command_kwargs = {}

                command_kwargs["short_help"] = "\n".join(numpy_doc["Summary"] or [])
                command_kwargs["help"] = "\n".join(numpy_doc["Extended Summary"] or [command_kwargs["short_help"]])

                # Now overwrite with any user supplied values
                command_kwargs.update(command_args)

                command = click.Command(name=command_name,
                                        callback=prepare_command()(command_callback),
                                        params=command_params,
                                        **command_kwargs)

                return command

            # Create the StageInfo
            stage_info = StageInfo(name=command_name,
                                   modes=modes,
                                   qualified_name=get_full_qualname(stage_class),
                                   build_command=build_command)

            # Save the stage information to be retrieved later
            stage_class._morpheus_registered_stage = stage_info

            existing_registrations: typing.Set[PipelineModes] = set()

            # Get any already registered nodes
            for mode in modes:
                registered_stage = GlobalStageRegistry.get().get_stage_info(command_name, mode)

                if (registered_stage is not None):

                    # Check if we were previously lazy registered
                    if (isinstance(registered_stage, LazyStageInfo)):
                        # Only check the qualified name
                        if (registered_stage.qualified_name != get_full_qualname(stage_class)):
                            raise RuntimeError(
                                f"Qualified name {registered_stage.qualified_name} != {get_full_qualname(stage_class)}")
                    elif (registered_stage != stage_info):
                        raise RuntimeError((
                            f"Registering stage '{command_name}' failed. Stage is already registered with different "
                            "options. Ensure `register_stage` is only executed once for each mode and name combination."
                            " If registered multiple times (i.e. on module reload), the registration must be identical"
                        ))

                    existing_registrations.update(registered_stage.modes)

            if (len(existing_registrations) > 0):
                diff_modes = existing_registrations.difference(modes)
                if (len(diff_modes) > 0):
                    raise RuntimeError("Mismatch between LazyStageInfo modes and register_stage() modes")
            else:
                # Not registered, add to global registry
                GlobalStageRegistry.get().add_stage_info(stage_info)

        return stage_class

    return register_stage_inner
