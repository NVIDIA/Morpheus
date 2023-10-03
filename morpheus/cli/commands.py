# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
"""Pipeline and tool subcommands for the Morpheus CLI."""

import functools
import logging
import os
import typing

import click

from morpheus.cli.plugin_manager import PluginManager
from morpheus.cli.stage_registry import GlobalStageRegistry
from morpheus.cli.stage_registry import LazyStageInfo
from morpheus.cli.utils import MorpheusRelativePath
from morpheus.cli.utils import get_config_from_ctx
from morpheus.cli.utils import get_enum_keys
from morpheus.cli.utils import get_log_levels
from morpheus.cli.utils import get_pipeline_from_ctx
from morpheus.cli.utils import load_labels_file
from morpheus.cli.utils import parse_enum
from morpheus.cli.utils import parse_log_level
from morpheus.cli.utils import prepare_command
from morpheus.config import AEFeatureScalar
from morpheus.config import Config
from morpheus.config import ConfigAutoEncoder
from morpheus.config import ConfigFIL
from morpheus.config import ConfigOnnxToTRT
from morpheus.config import CppConfig
from morpheus.config import PipelineModes
from morpheus.utils.logger import configure_logging

# pylint: disable=line-too-long, import-outside-toplevel, invalid-name, global-at-module-level, unused-argument

# NOTE: This file will get executed when hitting <TAB> for autocompletion. Any classes that require a long import time
# should be locally imported. For example, `morpheus.Pipeline` takes a long time to import and must be locally imported
# for each function

DEFAULT_CONFIG = Config()

# List all of the options in from morpheus.common.FileTypes without importing the object. This slows down
# autocomplete too much.
FILE_TYPE_NAMES = ["auto", "csv", "json"]

# Graphviz rankdir options ad documented in https://graphviz.org/docs/attr-types/rankdir/
RANKDIR_CHOICES = ['BT', 'LR', 'RL', 'TB']

ALIASES = {
    "pipeline": "pipeline-nlp",
}

global logger
logger = logging.getLogger("morpheus.cli")


# Command to add the command. We cache the response so this only executes once (which can happen with module load).
# `modes` is a tuple so it can be cached for LRU to work
@functools.lru_cache(maxsize=None)
def add_command(name: str, stage_module: str, modes: typing.Tuple[PipelineModes, ...] = None):
    """Add a stage to the global registry."""

    GlobalStageRegistry.get().add_stage_info(
        LazyStageInfo(name=name, stage_qualified_name=stage_module, modes=list(modes)))


class AliasedGroup(click.Group):
    """A click group that supports aliases."""

    def get_command(self, ctx, cmd_name):
        """Resolve aliases before calling the base method."""
        try:
            cmd_name = ALIASES[cmd_name]
        except KeyError:
            pass
        return super().get_command(ctx, cmd_name)


class PluginGroup(AliasedGroup):
    """Click group that loads subcommands from plugins."""

    def __init__(
        self,
        name: typing.Optional[str] = None,
        commands: typing.Optional[typing.Union[typing.Dict[str, click.Command], typing.Sequence[click.Command]]] = None,
        **attrs: typing.Any,
    ):
        self._pipeline_mode = attrs.pop("pipeline_mode", None)

        assert self._pipeline_mode is not None, "Must specify `pipeline_mode` when using `PluginGroup`"

        super().__init__(name, commands, **attrs)

        self._plugin_manager = PluginManager.get()

    def list_commands(self, ctx: click.Context) -> typing.List[str]:
        """Get the list of commands."""

        # Get the list of commands from the base
        command_list = set(super().list_commands(ctx))

        # Extend it with any plugins
        registered_stages = self._plugin_manager.get_registered_stages()

        plugin_command_list = registered_stages.get_registered_names(self._pipeline_mode)

        duplicate_commands = command_list.intersection(plugin_command_list)

        # The COMP_WORDS environment variable is set by click using the auto-complete feature, which may cause this
        # metho to be called multiple times.
        if (len(duplicate_commands) > 0 and 'COMP_WORDS' not in os.environ):
            raise RuntimeError(f"Plugins registered the following duplicate commands: {', '.join(duplicate_commands)}")

        command_list.update(plugin_command_list)

        command_list = sorted(command_list)

        return command_list

    def get_command(self, ctx, cmd_name):
        """Get the command."""

        # Check if the command is already loaded
        if (cmd_name not in self.commands):
            # Get the list of registered stages
            registered_stages = self._plugin_manager.get_registered_stages()

            stage_info = registered_stages.get_stage_info(cmd_name, self._pipeline_mode)

            # Build and save the command
            if (stage_info is not None):
                self.commands[cmd_name] = stage_info.build_command()
            else:
                # Save as none so we dont try again
                self.commands[cmd_name] = None

        return super().get_command(ctx, cmd_name)


@click.group(name="morpheus", chain=False, invoke_without_command=True, no_args_is_help=True, cls=AliasedGroup)
@click.option('--debug/--no-debug', default=False)
@click.option("--log_level",
              default=logging.getLevelName(DEFAULT_CONFIG.log_level),
              type=click.Choice(get_log_levels(), case_sensitive=False),
              callback=parse_log_level,
              help="Specify the logging level to use.")
@click.option('--log_config_file',
              default=DEFAULT_CONFIG.log_config_file,
              type=click.Path(exists=True, dir_okay=False),
              help=("Config file to use to configure logging. Use only for advanced situations. "
                    "Can accept both JSON and ini style configurations"))
@click.option("plugins",
              '--plugin',
              allow_from_autoenv=False,
              multiple=True,
              type=str,
              help=("Adds a Morpheus CLI plugin. "
                    "Can either be a module name or path to a python module"))
@click.version_option()
@prepare_command(parse_config=True)
def cli(ctx: click.Context,
        log_level: int = DEFAULT_CONFIG.log_level,
        log_config_file: str = DEFAULT_CONFIG.log_config_file,
        plugins: typing.List[str] = None,
        **kwargs):
    """Main entry point function for the CLI."""

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below
    ctx.ensure_object(dict)

    # Configure the logging
    configure_logging(log_level=log_level, log_config_file=log_config_file)

    # Re-get the logger class
    global logger  # pylint: disable=global-statement
    logger = logging.getLogger("morpheus.cli")

    if (plugins is not None):
        # If plugin is specified, add that to the plugin manager
        pm = PluginManager.get()

        for p in plugins:
            pm.add_plugin_option(p)


@cli.group(short_help="Run a utility tool", no_args_is_help=True)
@prepare_command()
def tools(ctx: click.Context, **kwargs):
    """Tools subcommand"""
    pass


@tools.command(short_help="Converts an ONNX model to a TRT engine")
@click.option("--input_model", type=click.Path(exists=True, readable=True), required=True)
@click.option("--output_model", type=click.Path(exists=False, writable=True), required=True)
@click.option('--batches', type=(int, int), required=True, multiple=True)
@click.option('--seq_length', type=int, required=True)
@click.option('--max_workspace_size', type=int, default=16000)
@prepare_command()
def onnx_to_trt(ctx: click.Context, **kwargs):
    """Converts an ONNX model to a TRT engine"""
    logger.info("Generating onnx file")

    # Convert batches to a list
    kwargs["batches"] = list(kwargs["batches"])

    c = ConfigOnnxToTRT()

    for (param, val) in kwargs.items():
        if hasattr(c, param):
            setattr(c, param, val)

    from morpheus.utils.onnx_to_trt import gen_engine

    gen_engine(c)


@tools.group(short_help="Utility for installing/updating/removing shell completion for Morpheus", no_args_is_help=True)
def autocomplete(**kwargs):
    """Utility for installing/updating/removing shell completion for Morpheus"""
    pass


@autocomplete.command(short_help="Show the Morpheus shell command completion code")
@click.option('--shell',
              required=False,
              type=click.Choice(["bash", "fish", "zsh"]),
              help="The shell to install completion to. Leave as the default to auto-detect")
def show(shell):
    """Show the click-completion-command completion code"""
    from morpheus.cli import click_completion_tools
    shell, path, code = click_completion_tools.get_code(shell=shell)

    click.secho(f"To add {shell} completion, write the following code to '{path}':\n", fg="blue")
    click.echo(code)


@autocomplete.command(short_help="Install the Morpheus shell command completion")
@click.option('--append/--overwrite', help="Append the completion code to the file", default=False)
@click.option('--shell',
              required=False,
              type=click.Choice(["bash", "fish", "zsh"]),
              help="The shell to install completion to. Leave as the default to auto-detect")
@click.option('--path',
              required=False,
              help="Location to install complete to. Leave empty to choose the default for the specified shell")
def install(**kwargs):
    """Install the click-completion-command completion"""
    from morpheus.cli import click_completion_tools
    shell, path = click_completion_tools.install_code(**kwargs)

    click.echo(f'{shell} completion installed in {path}')


@cli.group(short_help="Run one of the available pipelines", no_args_is_help=True, cls=AliasedGroup)
@click.option('--num_threads',
              default=os.cpu_count(),
              type=click.IntRange(min=1),
              help="Number of internal pipeline threads to use")
@click.option('--pipeline_batch_size',
              default=DEFAULT_CONFIG.pipeline_batch_size,
              type=click.IntRange(min=1),
              help=("Internal batch size for the pipeline. "
                    "Can be much larger than the model batch size. Also used for Kafka consumers"))
@click.option('--model_max_batch_size',
              default=DEFAULT_CONFIG.model_max_batch_size,
              type=click.IntRange(min=1),
              help="Max batch size to use for the model")
@click.option('--edge_buffer_size',
              default=DEFAULT_CONFIG.edge_buffer_size,
              type=click.IntRange(min=2),
              help=("The size of buffered channels to use between nodes in a pipeline. Larger values reduce "
                    "backpressure at the cost of memory. Smaller values will push messages through the "
                    "pipeline quicker. Must be greater than 1 and a power of 2 (i.e. 2, 4, 8, 16, etc.)"))
@click.option('--use_cpp',
              default=True,
              type=bool,
              help=("Whether or not to use C++ node and message types or to prefer python. "
                    "Only use as a last resort if bugs are encountered"))
@prepare_command(parse_config=True)
def run(ctx: click.Context, **kwargs):
    """Run subcommand, used for running a pipeline"""
    # Since the option isnt the same name as `should_use_cpp` anymore, manually set the value here.
    CppConfig.set_should_use_cpp(kwargs.pop("use_cpp", CppConfig.get_should_use_cpp()))

    pass


@click.group(chain=True,
             short_help="Run the inference pipeline with a NLP model",
             no_args_is_help=True,
             cls=PluginGroup,
             pipeline_mode=PipelineModes.NLP)
@click.option('--model_seq_length',
              default=256,
              type=click.IntRange(min=1),
              help=("Limits the length of the sequence returned. If tokenized string is shorter than max_length, "
                    "output will be padded with 0s. If the tokenized string is longer than max_length and "
                    "do_truncate == False, there will be multiple returned sequences containing the "
                    "overflowing token-ids. Default value is 256"))
@click.option('--label', type=str, default=None, multiple=True, help=("Specify output labels."))
@click.option('--labels_file',
              default="data/labels_nlp.txt",
              type=MorpheusRelativePath(dir_okay=False, exists=True, file_okay=True, resolve_path=True),
              help=("Specifies a file to read labels from in order to convert class IDs into labels. "
                    "A label file is a simple text file where each line corresponds to a label."
                    "Ignored when --label is specified"))
@click.option('--viz_file',
              default=None,
              type=click.Path(dir_okay=False, writable=True),
              help="Save a visualization of the pipeline at the specified location")
@click.option('--viz_direction',
              default="LR",
              type=click.Choice(RANKDIR_CHOICES, case_sensitive=False),
              help=("Set the direction for the Graphviz pipeline diagram, "
                    "ignored unless --viz_file is also specified."))
@prepare_command()
def pipeline_nlp(ctx: click.Context, **kwargs):
    """
    Configure and run the pipeline. To configure the pipeline, list the stages in the order that data should flow. The
    output of each stage will become the input for the next stage. For example, to read, classify and write to a file,
    the following stages could be used

    pipeline from-file --filename=my_dataset.json deserialize preprocess inf-triton --model_name=my_model
    --server_url=localhost:8001 filter --threshold=0.5 to-file --filename=classifications.json

    Pipelines must follow a few rules:
    1. Data must originate in a source stage. Current options are `from-file` or `from-kafka`
    2. A `deserialize` stage must be placed between the source stages and the rest of the pipeline
    3. Only one inference stage can be used. Zero is also fine
    4. The following stages must come after an inference stage: `add-class`, `filter`, `gen-viz`

    """

    click.secho("Configuring Pipeline via CLI", fg="green")

    config = get_config_from_ctx(ctx)
    config.mode = PipelineModes.NLP
    config.feature_length = kwargs["model_seq_length"]

    labels = kwargs['label']
    if len(labels):
        config.class_labels = list(labels)
    else:
        config.class_labels = load_labels_file(kwargs["labels_file"])
        logger.debug("Loaded labels file. Current labels: [%s]", str(config.class_labels))

    from morpheus.pipeline import LinearPipeline

    p = ctx.obj["pipeline"] = LinearPipeline(config)

    return p


@click.group(chain=True,
             short_help="Run the inference pipeline with a FIL model",
             no_args_is_help=True,
             cls=PluginGroup,
             pipeline_mode=PipelineModes.FIL)
@click.option('--model_fea_length',
              default=18,
              type=click.IntRange(min=1),
              help="Number of features trained in the model")
@click.option('--label',
              type=str,
              default=["mining"],
              multiple=True,
              help=("Specify output labels. Ignored when --labels_file is specified"))
@click.option('--labels_file',
              default=None,
              type=MorpheusRelativePath(dir_okay=False, exists=True, file_okay=True, resolve_path=True),
              help=("Specifies a file to read labels from in order to convert class IDs into labels. "
                    "A label file is a simple text file where each line corresponds to a label. "
                    "If unspecified the value specified by the --label flag will be used."))
@click.option('--columns_file',
              type=MorpheusRelativePath(dir_okay=False, exists=True, file_okay=True, resolve_path=True),
              help=("Specifies a file to read column features."))
@click.option('--viz_file',
              default=None,
              type=click.Path(dir_okay=False, writable=True),
              help="Save a visualization of the pipeline at the specified location")
@click.option('--viz_direction',
              default="LR",
              type=click.Choice(RANKDIR_CHOICES, case_sensitive=False),
              help=("Set the direction for the Graphviz pipeline diagram, "
                    "ignored unless --viz_file is also specified."))
@prepare_command()
def pipeline_fil(ctx: click.Context, **kwargs):
    """
    Configure and run the pipeline. To configure the pipeline, list the stages in the order that data should flow. The
    output of each stage will become the input for the next stage. For example, to read, classify and write to a file,
    the following stages could be used

    pipeline from-file --filename=my_dataset.json deserialize preprocess inf-triton --model_name=my_model
    --server_url=localhost:8001 filter --threshold=0.5 to-file --filename=classifications.json

    Pipelines must follow a few rules:
    1. Data must originate in a source stage. Current options are `from-file` or `from-kafka`
    2. A `deserialize` stage must be placed between the source stages and the rest of the pipeline
    3. Only one inference stage can be used. Zero is also fine
    4. The following stages must come after an inference stage: `add-class`, `filter`, `gen-viz`

    """

    click.secho("Configuring Pipeline via CLI", fg="green")

    config = get_config_from_ctx(ctx)
    config.mode = PipelineModes.FIL
    config.feature_length = kwargs["model_fea_length"]

    config.fil = ConfigFIL()

    labels_file = kwargs.get("labels_file")
    if (labels_file is not None):
        config.class_labels = load_labels_file(labels_file)
        logger.debug("Loaded labels file. Current labels: [%s]", str(config.class_labels))
    else:
        config.class_labels = list(kwargs['label'])

    if ("columns_file" in kwargs and kwargs["columns_file"] is not None):
        config.fil.feature_columns = load_labels_file(kwargs["columns_file"])
        logger.debug("Loaded columns. Current columns: [%s]", str(config.fil.feature_columns))

    from morpheus.pipeline import LinearPipeline

    p = ctx.obj["pipeline"] = LinearPipeline(config)

    return p


@click.group(chain=True,
             short_help="Run the inference pipeline with an AutoEncoder model",
             no_args_is_help=True,
             cls=PluginGroup,
             pipeline_mode=PipelineModes.AE)
@click.option('--columns_file',
              required=True,
              default=None,
              type=MorpheusRelativePath(dir_okay=False, exists=True, file_okay=True, resolve_path=True),
              help=("Specifies a file to read column features."))
@click.option('--labels_file',
              default=None,
              type=MorpheusRelativePath(dir_okay=False, exists=True, file_okay=True, resolve_path=True),
              help=("Specifies a file to read labels from in order to convert class IDs into labels. "
                    "A label file is a simple text file where each line corresponds to a label. "))
@click.option('--userid_column_name',
              type=str,
              default="userIdentityaccountId",
              required=True,
              help=("Which column to use as the User ID."))
@click.option('--userid_filter',
              type=str,
              default=None,
              help=("Specifying this value will filter all incoming data to only use rows with matching User IDs. "
                    "Which column is used for the User ID is specified by `userid_column_name`"))
@click.option('--feature_scaler',
              type=click.Choice(get_enum_keys(AEFeatureScalar), case_sensitive=False),
              default=AEFeatureScalar.STANDARD.name,
              callback=functools.partial(parse_enum, enum_class=AEFeatureScalar, case_sensitive=False),
              help=("Autoencoder feature scaler"))
@click.option('--use_generic_model',
              is_flag=True,
              type=bool,
              help=("Whether to use a generic model when user does not have minimum number of training rows"))
@click.option('--viz_file',
              default=None,
              type=click.Path(dir_okay=False, writable=True),
              help="Save a visualization of the pipeline at the specified location")
@click.option('--viz_direction',
              default="LR",
              type=click.Choice(RANKDIR_CHOICES, case_sensitive=False),
              help=("Set the direction for the Graphviz pipeline diagram, "
                    "ignored unless --viz_file is also specified."))
@prepare_command()
def pipeline_ae(ctx: click.Context, **kwargs):
    """
    Configure and run the pipeline. To configure the pipeline, list the stages in the order that data should flow. The
    output of each stage will become the input for the next stage. For example, to read, classify and write to a file,
    the following stages could be used

    pipeline from-file --filename=my_dataset.json deserialize preprocess inf-triton --model_name=my_model
    --server_url=localhost:8001 filter --threshold=0.5 to-file --filename=classifications.json

    Pipelines must follow a few rules:
    1. Data must originate in a source stage. Current options are `from-file` or `from-kafka`
    2. A `deserialize` stage must be placed between the source stages and the rest of the pipeline
    3. Only one inference stage can be used. Zero is also fine
    4. The following stages must come after an inference stage: `add-class`, `filter`, `gen-viz`

    """

    click.secho("Configuring Pipeline via CLI", fg="green")

    config = get_config_from_ctx(ctx)
    config.mode = PipelineModes.AE

    if CppConfig.get_should_use_cpp():
        logger.warning("C++ is disabled for AutoEncoder pipelines at this time.")
        CppConfig.set_should_use_cpp(False)

    config.ae = ConfigAutoEncoder()
    config.ae.userid_column_name = kwargs["userid_column_name"]
    config.ae.feature_scaler = kwargs["feature_scaler"]
    config.ae.use_generic_model = kwargs["use_generic_model"]
    config.ae.feature_columns = load_labels_file(kwargs["columns_file"])
    logger.debug("Loaded columns. Current columns: [%s]", str(config.ae.feature_columns))

    if ("labels_file" in kwargs and kwargs["labels_file"] is not None):
        config.class_labels = load_labels_file(kwargs["labels_file"])
        logger.debug("Loaded labels file. Current labels: [%s]", str(config.class_labels))
    else:
        # Use default labels
        config.class_labels = ["reconstruct_loss", "zscore"]

    if ("userid_filter" in kwargs):
        config.ae.userid_filter = kwargs["userid_filter"]

        logger.info("Filtering all users except ID: '%s'", str(config.ae.userid_filter))

    from morpheus.pipeline import LinearPipeline

    p = ctx.obj["pipeline"] = LinearPipeline(config)

    return p


@click.group(chain=True,
             short_help="Run a custom inference pipeline without a specific model type",
             no_args_is_help=True,
             cls=PluginGroup,
             pipeline_mode=PipelineModes.OTHER)
@click.option('--model_fea_length',
              default=1,
              type=click.IntRange(min=1),
              help="Number of features trained in the model")
@click.option('--label',
              type=str,
              default=None,
              multiple=True,
              help=("Specify output labels. Ignored when --labels_file is specified"))
@click.option('--labels_file',
              default=None,
              type=MorpheusRelativePath(dir_okay=False, exists=True, file_okay=True, resolve_path=True),
              help=("Specifies a file to read labels from in order to convert class IDs into labels. "
                    "A label file is a simple text file where each line corresponds to a label."))
@click.option('--viz_file',
              default=None,
              type=click.Path(dir_okay=False, writable=True),
              help="Save a visualization of the pipeline at the specified location")
@click.option('--viz_direction',
              default="LR",
              type=click.Choice(RANKDIR_CHOICES, case_sensitive=False),
              help=("Set the direction for the Graphviz pipeline diagram, "
                    "ignored unless --viz_file is also specified."))
@prepare_command()
def pipeline_other(ctx: click.Context, **kwargs):
    """
    Configure and run the pipeline. To configure the pipeline, list the stages in the order that data should flow. The
    output of each stage will become the input for the next stage. For example, to read, classify and write to a file,
    the following stages could be used

    pipeline from-file --filename=my_dataset.json deserialize preprocess inf-triton --model_name=my_model
    --server_url=localhost:8001 filter --threshold=0.5 to-file --filename=classifications.json

    Pipelines must follow a few rules:
    1. Data must originate in a source stage. Current options are `from-file` or `from-kafka`
    2. A `deserialize` stage must be placed between the source stages and the rest of the pipeline
    3. Only one inference stage can be used. Zero is also fine
    4. The following stages must come after an inference stage: `add-class`, `filter`, `gen-viz`

    """

    click.secho("Configuring Pipeline via CLI", fg="green")

    config = get_config_from_ctx(ctx)
    config.mode = PipelineModes.OTHER
    config.feature_length = kwargs["model_fea_length"]

    config.fil = ConfigFIL()

    labels_file = kwargs.get("labels_file")
    if (labels_file is not None):
        config.class_labels = load_labels_file(labels_file)
        logger.debug("Loaded labels file. Current labels: [%s]", str(config.class_labels))
    else:
        labels = kwargs["label"]
        if len(labels):
            config.class_labels = list(labels)

    from morpheus.pipeline import LinearPipeline

    p = ctx.obj["pipeline"] = LinearPipeline(config)

    return p


@pipeline_nlp.result_callback()
@pipeline_fil.result_callback()
@pipeline_ae.result_callback()
@pipeline_other.result_callback()
@click.pass_context
def post_pipeline(ctx: click.Context, *args, **kwargs):
    """Executes the pipeline"""

    config = get_config_from_ctx(ctx)

    logger.info("Config: \n%s", config.to_string())
    logger.info("CPP Enabled: %s", CppConfig.get_should_use_cpp())

    click.secho("Starting pipeline via CLI... Ctrl+C to Quit", fg="red")

    pipeline = get_pipeline_from_ctx(ctx)

    # Run the pipeline before generating visualization to ensure the pipeline has been started
    pipeline.run()

    # TODO(MDD): Move visualization before `pipeline.run()` once Issue #230 is fixed.
    if ("viz_file" in kwargs and kwargs["viz_file"] is not None):
        pipeline.visualize(kwargs["viz_file"], rankdir=kwargs["viz_direction"].upper())
        click.secho(f"Pipeline visualization saved to {kwargs['viz_file']}", fg="yellow")


# Manually create the subcommands for each command (necessary since commands can be used on multiple groups)
run.add_command(pipeline_nlp)
run.add_command(pipeline_fil)
run.add_command(pipeline_ae)
run.add_command(pipeline_other)

ALL = (PipelineModes.AE, PipelineModes.NLP, PipelineModes.FIL, PipelineModes.OTHER)
NOT_AE = (PipelineModes.NLP, PipelineModes.FIL, PipelineModes.OTHER)
AE_ONLY = (PipelineModes.AE, )
FIL_ONLY = (PipelineModes.FIL, )
NLP_ONLY = (PipelineModes.NLP, )

# Keep these sorted!!!
add_command("add-class", "morpheus.stages.postprocess.add_classifications_stage.AddClassificationsStage", modes=ALL)
add_command("add-scores", "morpheus.stages.postprocess.add_scores_stage.AddScoresStage", modes=ALL)
add_command("buffer", "morpheus.stages.general.buffer_stage.BufferStage", modes=ALL)
add_command("delay", "morpheus.stages.general.delay_stage.DelayStage", modes=ALL)
add_command("deserialize", "morpheus.stages.preprocess.deserialize_stage.DeserializeStage", modes=NOT_AE)
add_command("dropna", "morpheus.stages.preprocess.drop_null_stage.DropNullStage", modes=NOT_AE)
add_command("file-source", "morpheus.stages.input.file_source.FileSource", modes=NOT_AE)
add_command("filter", "morpheus.stages.postprocess.filter_detections_stage.FilterDetectionsStage", modes=ALL)
add_command("from-azure", "morpheus.stages.input.azure_source_stage.AzureSourceStage", modes=AE_ONLY)
add_command("from-appshield", "morpheus.stages.input.appshield_source_stage.AppShieldSourceStage", modes=FIL_ONLY)
add_command("from-azure", "morpheus.stages.input.azure_source_stage.AzureSourceStage", modes=AE_ONLY)
add_command("from-cloudtrail", "morpheus.stages.input.cloud_trail_source_stage.CloudTrailSourceStage", modes=AE_ONLY)
add_command("from-duo", "morpheus.stages.input.duo_source_stage.DuoSourceStage", modes=AE_ONLY)
add_command("from-file", "morpheus.stages.input.file_source_stage.FileSourceStage", modes=NOT_AE)
add_command("from-kafka", "morpheus.stages.input.kafka_source_stage.KafkaSourceStage", modes=NOT_AE)
add_command("from-http", "morpheus.stages.input.http_server_source_stage.HttpServerSourceStage", modes=ALL)
add_command("from-http-client", "morpheus.stages.input.http_client_source_stage.HttpClientSourceStage", modes=ALL)
add_command("from-rss", "morpheus.stages.input.rss_source_stage.RSSSourceStage", modes=ALL)
add_command("gen-viz", "morpheus.stages.postprocess.generate_viz_frames_stage.GenerateVizFramesStage", modes=NLP_ONLY)
add_command("inf-identity", "morpheus.stages.inference.identity_inference_stage.IdentityInferenceStage", modes=NOT_AE)
add_command("inf-pytorch",
            "morpheus.stages.inference.auto_encoder_inference_stage.AutoEncoderInferenceStage",
            modes=AE_ONLY)
add_command("inf-pytorch", "morpheus.stages.inference.pytorch_inference_stage.PyTorchInferenceStage", modes=NOT_AE)
add_command("inf-triton", "morpheus.stages.inference.triton_inference_stage.TritonInferenceStage", modes=ALL)
add_command("mlflow-drift", "morpheus.stages.postprocess.ml_flow_drift_stage.MLFlowDriftStage", modes=NOT_AE)
add_command("monitor", "morpheus.stages.general.monitor_stage.MonitorStage", modes=ALL)
add_command("preprocess", "morpheus.stages.preprocess.preprocess_ae_stage.PreprocessAEStage", modes=AE_ONLY)
add_command("preprocess", "morpheus.stages.preprocess.preprocess_fil_stage.PreprocessFILStage", modes=FIL_ONLY)
add_command("preprocess", "morpheus.stages.preprocess.preprocess_nlp_stage.PreprocessNLPStage", modes=NLP_ONLY)
add_command("serialize", "morpheus.stages.postprocess.serialize_stage.SerializeStage", modes=ALL)
add_command("timeseries", "morpheus.stages.postprocess.timeseries_stage.TimeSeriesStage", modes=AE_ONLY)
add_command("to-elasticsearch",
            "morpheus.stages.output.write_to_elasticsearch_stage.WriteToElasticsearchStage",
            modes=ALL)
add_command("to-file", "morpheus.stages.output.write_to_file_stage.WriteToFileStage", modes=ALL)
add_command("to-kafka", "morpheus.stages.output.write_to_kafka_stage.WriteToKafkaStage", modes=ALL)
add_command("to-http", "morpheus.stages.output.http_client_sink_stage.HttpClientSinkStage", modes=ALL)
add_command("to-http-server", "morpheus.stages.output.http_server_sink_stage.HttpServerSinkStage", modes=ALL)
add_command("train-ae", "morpheus.stages.preprocess.train_ae_stage.TrainAEStage", modes=AE_ONLY)
add_command("trigger", "morpheus.stages.general.trigger_stage.TriggerStage", modes=ALL)
add_command("validate", "morpheus.stages.postprocess.validation_stage.ValidationStage", modes=ALL)

add_command("from-doca", "morpheus.stages.doca.doca_source_stage.DocaSourceStage", modes=NLP_ONLY)
