#!/usr/bin/env python
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

import os
import shutil
import warnings
from unittest import mock

import click
import mlflow
import pytest
from click.testing import CliRunner
from mlflow.tracking import fluent

import morpheus
from _utils import TEST_DIRS
from _utils.stages.conv_msg import ConvMsg
from morpheus.cli import commands
from morpheus.common import FileTypes
from morpheus.common import FilterSource
from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.stages.general.monitor_stage import MonitorStage
from morpheus.stages.inference.identity_inference_stage import IdentityInferenceStage
from morpheus.stages.inference.pytorch_inference_stage import PyTorchInferenceStage
from morpheus.stages.inference.triton_inference_stage import TritonInferenceStage
from morpheus.stages.input.file_source_stage import FileSourceStage
from morpheus.stages.input.kafka_source_stage import KafkaSourceStage
from morpheus.stages.output.write_to_file_stage import WriteToFileStage
from morpheus.stages.output.write_to_kafka_stage import WriteToKafkaStage
from morpheus.stages.postprocess.add_classifications_stage import AddClassificationsStage
from morpheus.stages.postprocess.add_scores_stage import AddScoresStage
from morpheus.stages.postprocess.filter_detections_stage import FilterDetectionsStage
from morpheus.stages.postprocess.ml_flow_drift_stage import MLFlowDriftStage
from morpheus.stages.postprocess.serialize_stage import SerializeStage
from morpheus.stages.postprocess.validation_stage import ValidationStage
from morpheus.stages.preprocess.deserialize_stage import DeserializeStage
from morpheus.stages.preprocess.drop_null_stage import DropNullStage
from morpheus.stages.preprocess.preprocess_fil_stage import PreprocessFILStage
from morpheus.stages.preprocess.preprocess_nlp_stage import PreprocessNLPStage
from morpheus.utils.file_utils import load_labels_file

GENERAL_ARGS = ['run', '--num_threads=12', '--pipeline_batch_size=1024', '--model_max_batch_size=1024', '--use_cpp=0']
MONITOR_ARGS = ['monitor', '--description', 'Unittest', '--smoothing=0.001', '--unit', 'inf']
VALIDATE_ARGS = [
    'validate',
    '--val_file_name',
    os.path.join(TEST_DIRS.validation_data_dir, 'dfp-cloudtrail-role-g-validation-data-output.csv'),
    '--results_file_name=results.json',
    '--index_col=_index_',
    '--exclude',
    'event_dt',
    '--rel_tol=0.1'
]
TO_FILE_ARGS = ['to-file', '--filename=out.csv']
FILE_SRC_ARGS = [
    'from-file', '--filename', os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
]
INF_TRITON_ARGS = ['inf-triton', '--model_name=test-model', '--server_url=test:123', '--force_convert_inputs=True']

KAFKA_BOOTS = ['--bootstrap_servers', 'kserv1:123,kserv2:321']
FROM_KAFKA_ARGS = ['from-kafka', '--input_topic', 'test_topic'] + KAFKA_BOOTS
TO_KAFKA_ARGS = ['to-kafka', '--output_topic', 'test_topic'] + KAFKA_BOOTS

# Pylint doesn't understand how pytest fixtures work and flags fixture uasage as a redefinition of the symbol in the
# outer scope.
# pylint: disable=redefined-outer-name


# Fixtures specific to the cli tests
@pytest.fixture(scope="function")
def callback_values(request: pytest.FixtureRequest):
    """
    Replaces the results_callback in cli which executes the pipeline.
    Allowing us to examine/verify that cli built us a propper pipeline
    without actually running it. When run the callback will update the
    `callback_values` dictionary with the context, pipeline & stages constructed.
    """
    callback_values_ = {}

    marker = request.node.get_closest_marker("replace_callback")
    group_name = marker.args[0]
    group = getattr(commands, group_name)

    @group.result_callback(replace=True)
    @click.pass_context
    def mock_post_callback(ctx, stages, *a, **k):  # pylint: disable=unused-argument
        callback_values_.update({'ctx': ctx, 'stages': stages, 'pipe': ctx.obj["pipeline"]})
        ctx.exit(47)

    return callback_values_


@pytest.fixture(scope="function")
def mlflow_uri(tmp_path):
    experiment_name = "Morpheus"
    uri = f"file://{tmp_path}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    with mlflow.start_run(run_name="Model Drift",
                          tags={"morpheus.type": "drift"},
                          experiment_id=experiment.experiment_id):
        pass

    yield uri

    num_runs = len(fluent._active_run_stack)
    for _ in range(num_runs):
        mlflow.end_run()


@pytest.fixture(scope="function", autouse=True)
def config_warning_fixture():
    # morpheus.cli.utils._apply_to_config method will warn about any keyword arguments that don't match a config option
    # this isn't triggered in normal production code, but is triggered in the cli tests.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No config option matches for.*", category=UserWarning)
        yield


@pytest.mark.reload_modules(commands)
@pytest.mark.usefixtures("chdir_tmpdir", "reload_modules")
class TestCLI:

    @pytest.mark.parametrize(
        'cmd', [[], ['tools'], ['run'], ['run', 'pipeline-fil'], ['run', 'pipeline-nlp'], ['run', 'pipeline-other']])
    def test_help(self, cmd: list[str]):
        runner = CliRunner()
        result = runner.invoke(commands.cli, cmd + ['--help'])
        assert result.exit_code == 0, result.output

    @pytest.mark.parametrize('cmd',
                             [['tools', 'autocomplete', 'show'], ['tools', 'autocomplete', 'install', '--shell=bash']])
    def test_autocomplete(self, tmp_path, cmd: list[str]):
        runner = CliRunner()
        result = runner.invoke(commands.cli, cmd, env={'HOME': str(tmp_path)})
        assert result.exit_code == 0, result.output

    @pytest.mark.usefixtures("restore_environ")
    @pytest.mark.parametrize('use_environ', [True, False])
    @pytest.mark.parametrize('value', [1, 13, 33])
    @mock.patch('morpheus.utils.seed.manual_seed')
    def test_manual_seed(self, mock_manual_seed: mock.MagicMock, value: int, use_environ: bool):
        flags = ['run']
        if use_environ:
            os.environ['MORPHEUS_MANUAL_SEED'] = str(value)
        else:
            flags.append(f'--manual_seed={value}')

        flags.append('pipeline-other')

        runner = CliRunner()
        result = runner.invoke(commands.cli, flags)
        assert result.exit_code == 0, result.output
        mock_manual_seed.assert_called_once_with(value)

    @pytest.mark.replace_callback('pipeline_fil')
    def test_pipeline_fil(self, config, callback_values):
        """
        Creates a pipeline roughly matching that of the abp validation test
        """
        args = (GENERAL_ARGS + ['pipeline-fil', '--columns_file=data/columns_fil.txt'] + FILE_SRC_ARGS +
                ['deserialize', 'preprocess'] + INF_TRITON_ARGS + MONITOR_ARGS + ['add-class'] + VALIDATE_ARGS +
                ['serialize'] + TO_FILE_ARGS)

        obj = {}
        runner = CliRunner()
        result = runner.invoke(commands.cli, args, obj=obj)
        assert result.exit_code == 47, result.output

        # Ensure our config is populated correctly
        config = obj["config"]
        assert config.mode == PipelineModes.FIL
        assert config.class_labels == ["mining"]

        expected_columns = load_labels_file(os.path.join(TEST_DIRS.data_dir, 'columns_fil.txt'))
        assert config.fil.feature_columns == expected_columns

        assert config.ae is None

        pipe = callback_values['pipe']
        assert pipe is not None

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [file_source, deserialize, process_fil, triton_inf, monitor, add_class, validation, serialize, to_file] = stages

        assert isinstance(file_source, FileSourceStage)
        assert file_source._filename == os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
        assert not file_source._iterative

        assert isinstance(deserialize, DeserializeStage)
        assert isinstance(process_fil, PreprocessFILStage)

        assert isinstance(triton_inf, TritonInferenceStage)
        assert triton_inf._model_name == 'test-model'
        assert triton_inf._server_url == 'test:123'
        assert triton_inf._force_convert_inputs

        assert isinstance(monitor, MonitorStage)
        assert monitor._mc._description == 'Unittest'
        assert monitor._mc._smoothing == 0.001
        assert monitor._mc._unit == 'inf'

        assert isinstance(add_class, AddClassificationsStage)

        assert isinstance(validation, ValidationStage)
        assert validation._results_file_name == 'results.json'
        assert validation._index_col == '_index_'

        # Click appears to be converting this into a tuple
        assert list(validation._exclude_columns) == ['event_dt']
        assert validation._rel_tol == 0.1

        assert isinstance(serialize, SerializeStage)
        assert isinstance(to_file, WriteToFileStage)
        assert to_file._controller._output_file == 'out.csv'

    @pytest.mark.replace_callback('pipeline_fil')
    def test_pipeline_fil_all(self, config, callback_values, tmp_path, mlflow_uri):
        """
        Attempt to add all possible stages to the pipeline_fil, even if the pipeline doesn't
        actually make sense, just test that cli could assemble it
        """
        tmp_model = os.path.join(tmp_path, 'fake-model.file')
        with open(tmp_model, 'w', encoding='UTF-8') as fh:
            pass

        labels_file = os.path.join(tmp_path, 'labels.txt')
        with open(labels_file, 'w', encoding='UTF-8') as fh:
            fh.writelines(['frogs\n', 'lizards\n', 'toads'])

        args = (GENERAL_ARGS + ['pipeline-fil', '--labels_file', labels_file, '--columns_file=data/columns_fil.txt'] +
                FILE_SRC_ARGS + FROM_KAFKA_ARGS + [
                    'deserialize',
                    'filter',
                    'dropna',
                    '--column',
                    'xyz',
                    'preprocess',
                    'add-scores',
                    'unittest-conv-msg',
                    'inf-identity',
                    'inf-pytorch',
                    '--model_filename',
                    tmp_model,
                    'mlflow-drift',
                    '--tracking_uri',
                    mlflow_uri
                ] + INF_TRITON_ARGS + MONITOR_ARGS + ['add-class'] + VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS +
                TO_KAFKA_ARGS)

        obj = {}
        runner = CliRunner()
        result = runner.invoke(commands.cli, args, obj=obj)
        assert result.exit_code == 47, result.output

        # Ensure our config is populated correctly

        config = obj["config"]
        assert config.mode == PipelineModes.FIL
        assert config.class_labels == ['frogs', 'lizards', 'toads']

        assert config.ae is None

        pipe = callback_values['pipe']
        assert pipe is not None

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [
            file_source,
            from_kafka,
            deserialize,
            filter_stage,
            dropna,
            process_fil,
            add_scores,
            conv_msg,
            inf_ident,
            inf_pytorch,
            mlflow_drift,
            triton_inf,
            monitor,
            add_class,
            validation,
            serialize,
            to_file,
            to_kafka
        ] = stages

        assert isinstance(file_source, FileSourceStage)
        assert file_source._filename == os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
        assert not file_source._iterative

        assert isinstance(from_kafka, KafkaSourceStage)
        assert from_kafka._consumer_params['bootstrap.servers'] == 'kserv1:123,kserv2:321'
        assert from_kafka._topics == ['test_topic']

        assert isinstance(deserialize, DeserializeStage)
        assert isinstance(filter_stage, FilterDetectionsStage)

        assert isinstance(dropna, DropNullStage)
        assert dropna._column == 'xyz'

        assert isinstance(process_fil, PreprocessFILStage)

        assert isinstance(add_scores, AddScoresStage)
        assert isinstance(conv_msg, ConvMsg)
        assert isinstance(inf_ident, IdentityInferenceStage)

        assert isinstance(inf_pytorch, PyTorchInferenceStage)
        assert inf_pytorch._model_filename == tmp_model

        assert isinstance(mlflow_drift, MLFlowDriftStage)
        assert mlflow_drift._tracking_uri == mlflow_uri

        assert isinstance(triton_inf, TritonInferenceStage)
        assert triton_inf._model_name == 'test-model'
        assert triton_inf._server_url == 'test:123'
        assert triton_inf._force_convert_inputs

        assert isinstance(monitor, MonitorStage)
        assert monitor._mc._description == 'Unittest'
        assert monitor._mc._smoothing == 0.001
        assert monitor._mc._unit == 'inf'

        assert isinstance(add_class, AddClassificationsStage)

        assert isinstance(validation, ValidationStage)
        assert validation._results_file_name == 'results.json'
        assert validation._index_col == '_index_'

        # Click appears to be converting this into a tuple
        assert list(validation._exclude_columns) == ['event_dt']
        assert validation._rel_tol == 0.1

        assert isinstance(serialize, SerializeStage)

        assert isinstance(to_file, WriteToFileStage)
        assert to_file._controller._output_file == 'out.csv'

        assert isinstance(to_kafka, WriteToKafkaStage)
        assert to_kafka._kafka_conf['bootstrap.servers'] == 'kserv1:123,kserv2:321'
        assert to_kafka._output_topic == 'test_topic'

    @pytest.mark.replace_callback('pipeline_other')
    def test_enum_parsing(self, config, callback_values, tmp_path, mlflow_uri):
        """
        Test parsing of CLI flags for C++ enum values issue #675
        """
        tmp_model = os.path.join(tmp_path, 'fake-model.file')
        with open(tmp_model, 'w', encoding='UTF-8'):
            pass

        labels_file = os.path.join(tmp_path, 'labels.txt')
        with open(labels_file, 'w', encoding='UTF-8') as fh:
            fh.writelines(['frogs\n', 'lizards\n', 'toads'])

        file_src_args = FILE_SRC_ARGS[:]
        file_src_args.append('--file_type=json')

        from_kafka_args = FROM_KAFKA_ARGS[:]
        from_kafka_args.append('--auto_offset_reset=earliest')

        to_file_args = TO_FILE_ARGS[:]
        to_file_args.append('--file_type=csv')

        args = (GENERAL_ARGS + ['pipeline-other', '--labels_file', labels_file] + file_src_args + from_kafka_args + [
            'deserialize',
            'filter',
            '--filter_source=tensor',
            'dropna',
            '--column',
            'xyz',
            'add-scores',
            'unittest-conv-msg',
            'inf-identity',
            'inf-pytorch',
            '--model_filename',
            tmp_model,
            'mlflow-drift',
            '--tracking_uri',
            mlflow_uri
        ] + INF_TRITON_ARGS + MONITOR_ARGS + ['add-class'] + VALIDATE_ARGS + ['serialize'] + to_file_args +
                TO_KAFKA_ARGS)

        obj = {}
        runner = CliRunner()
        result = runner.invoke(commands.cli, args, obj=obj)
        assert result.exit_code == 47, result.output

        # Ensure our config is populated correctly

        config = obj["config"]
        assert config.mode == PipelineModes.OTHER

        assert config.ae is None

        pipe = callback_values['pipe']
        assert pipe is not None

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [
            file_source,
            from_kafka,
            deserialize,
            filter_stage,
            dropna,
            add_scores,
            conv_msg,
            inf_ident,
            inf_pytorch,
            mlflow_drift,
            triton_inf,
            monitor,
            add_class,
            validation,
            serialize,
            to_file,
            to_kafka
        ] = stages

        assert isinstance(file_source, FileSourceStage)
        assert file_source._filename == os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
        assert not file_source._iterative
        assert file_source._file_type == FileTypes.JSON

        assert isinstance(from_kafka, KafkaSourceStage)
        assert from_kafka._consumer_params['bootstrap.servers'] == 'kserv1:123,kserv2:321'
        assert from_kafka._topics == ['test_topic']
        assert from_kafka._consumer_params['auto.offset.reset'] == 'earliest'

        assert isinstance(deserialize, DeserializeStage)
        assert isinstance(filter_stage, FilterDetectionsStage)
        assert filter_stage._controller._filter_source == FilterSource.TENSOR

        assert isinstance(dropna, DropNullStage)
        assert dropna._column == 'xyz'

        assert isinstance(add_scores, AddScoresStage)
        assert isinstance(conv_msg, ConvMsg)
        assert isinstance(inf_ident, IdentityInferenceStage)

        assert isinstance(inf_pytorch, PyTorchInferenceStage)
        assert inf_pytorch._model_filename == tmp_model

        assert isinstance(mlflow_drift, MLFlowDriftStage)
        assert mlflow_drift._tracking_uri == mlflow_uri

        assert isinstance(triton_inf, TritonInferenceStage)
        assert triton_inf._model_name == 'test-model'
        assert triton_inf._server_url == 'test:123'
        assert triton_inf._force_convert_inputs

        assert isinstance(monitor, MonitorStage)
        assert monitor._mc._description == 'Unittest'
        assert monitor._mc._smoothing == 0.001
        assert monitor._mc._unit == 'inf'

        assert isinstance(add_class, AddClassificationsStage)

        assert isinstance(validation, ValidationStage)
        assert validation._results_file_name == 'results.json'
        assert validation._index_col == '_index_'

        # Click appears to be converting this into a tuple
        assert list(validation._exclude_columns) == ['event_dt']
        assert validation._rel_tol == 0.1

        assert isinstance(serialize, SerializeStage)

        assert isinstance(to_file, WriteToFileStage)
        assert to_file._controller._output_file == 'out.csv'
        assert to_file._controller._file_type == FileTypes.CSV

        assert isinstance(to_kafka, WriteToKafkaStage)
        assert to_kafka._kafka_conf['bootstrap.servers'] == 'kserv1:123,kserv2:321'
        assert to_kafka._output_topic == 'test_topic'

    @pytest.mark.replace_callback('pipeline_nlp')
    def test_pipeline_nlp(self, config, callback_values):
        """
        Build a pipeline roughly ressembles the phishing validation script
        """
        labels_file = os.path.join(TEST_DIRS.data_dir, 'labels_phishing.txt')
        vocab_file_name = os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt')
        args = (GENERAL_ARGS + ['pipeline-nlp', '--model_seq_length=128', '--labels_file', labels_file] +
                FILE_SRC_ARGS + [
                    'deserialize',
                    'preprocess',
                    '--vocab_hash_file',
                    vocab_file_name,
                    '--truncation=True',
                    '--do_lower_case=True',
                    '--add_special_tokens=False'
                ] + INF_TRITON_ARGS + MONITOR_ARGS + ['add-class', '--label=is_phishing', '--threshold=0.7'] +
                VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS)

        obj = {}
        runner = CliRunner()
        result = runner.invoke(commands.cli, args, obj=obj)
        assert result.exit_code == 47, result.output

        # Ensure our config is populated correctly
        config = obj["config"]
        assert config.mode == PipelineModes.NLP
        assert config.class_labels == ["not_phishing", "is_phishing"]
        assert config.feature_length == 128

        assert config.ae is None

        pipe = callback_values['pipe']
        assert pipe is not None

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [file_source, deserialize, process_nlp, triton_inf, monitor, add_class, validation, serialize, to_file] = stages

        assert isinstance(file_source, FileSourceStage)
        assert file_source._filename == os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
        assert not file_source._iterative

        assert isinstance(deserialize, DeserializeStage)

        assert isinstance(process_nlp, PreprocessNLPStage)
        assert process_nlp._vocab_hash_file == os.path.realpath(vocab_file_name)
        assert process_nlp._truncation
        assert process_nlp._do_lower_case
        assert not process_nlp._add_special_tokens

        assert isinstance(triton_inf, TritonInferenceStage)
        assert triton_inf._model_name == 'test-model'
        assert triton_inf._server_url == 'test:123'
        assert triton_inf._force_convert_inputs

        assert isinstance(monitor, MonitorStage)
        assert monitor._mc._description == 'Unittest'
        assert monitor._mc._smoothing == 0.001
        assert monitor._mc._unit == 'inf'

        assert isinstance(add_class, AddClassificationsStage)
        assert add_class._labels == ('is_phishing', )
        assert add_class._threshold == 0.7

        assert isinstance(validation, ValidationStage)
        assert validation._results_file_name == 'results.json'
        assert validation._index_col == '_index_'

        # Click appears to be converting this into a tuple
        assert list(validation._exclude_columns) == ['event_dt']
        assert validation._rel_tol == 0.1

        assert isinstance(serialize, SerializeStage)

        assert isinstance(to_file, WriteToFileStage)
        assert to_file._controller._output_file == 'out.csv'

    @pytest.mark.replace_callback('pipeline_nlp')
    def test_pipeline_nlp_all(self, config, callback_values, tmp_path, mlflow_uri):
        """
        Attempt to add all possible stages to the pipeline_nlp, even if the pipeline doesn't
        actually make sense, just test that cli could assemble it
        """
        tmp_model = os.path.join(tmp_path, 'fake-model.file')
        with open(tmp_model, 'w', encoding='UTF-8'):
            pass

        labels_file = os.path.join(TEST_DIRS.data_dir, 'labels_phishing.txt')
        vocab_file_name = os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt')
        args = (GENERAL_ARGS + ['pipeline-nlp', '--model_seq_length=128', '--labels_file', labels_file] +
                FILE_SRC_ARGS + FROM_KAFKA_ARGS + [
                    'deserialize',
                    'filter',
                    'dropna',
                    '--column',
                    'xyz',
                    'preprocess',
                    '--vocab_hash_file',
                    vocab_file_name,
                    '--truncation=True',
                    '--do_lower_case=True',
                    '--add_special_tokens=False',
                    'add-scores',
                    'unittest-conv-msg',
                    'inf-identity',
                    'inf-pytorch',
                    '--model_filename',
                    tmp_model,
                    'mlflow-drift',
                    '--tracking_uri',
                    mlflow_uri
                ] + INF_TRITON_ARGS + MONITOR_ARGS + ['add-class', '--label=is_phishing', '--threshold=0.7'] +
                VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS + TO_KAFKA_ARGS)

        obj = {}
        runner = CliRunner()
        result = runner.invoke(commands.cli, args, obj=obj)
        assert result.exit_code == 47, result.output

        # Ensure our config is populated correctly
        config = obj["config"]
        assert config.mode == PipelineModes.NLP
        assert config.class_labels == ["not_phishing", "is_phishing"]
        assert config.feature_length == 128

        assert config.ae is None

        pipe = callback_values['pipe']
        assert pipe is not None

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [
            file_source,
            from_kafka,
            deserialize,
            filter_stage,
            dropna,
            process_nlp,
            add_scores,
            conv_msg,
            inf_ident,
            inf_pytorch,
            mlflow_drift,
            triton_inf,
            monitor,
            add_class,
            validation,
            serialize,
            to_file,
            to_kafka
        ] = stages

        assert isinstance(file_source, FileSourceStage)
        assert file_source._filename == os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
        assert not file_source._iterative

        assert isinstance(from_kafka, KafkaSourceStage)
        assert from_kafka._consumer_params['bootstrap.servers'] == 'kserv1:123,kserv2:321'
        assert from_kafka._topics == ['test_topic']

        assert isinstance(deserialize, DeserializeStage)
        assert isinstance(filter_stage, FilterDetectionsStage)

        assert isinstance(dropna, DropNullStage)
        assert dropna._column == 'xyz'

        assert isinstance(process_nlp, PreprocessNLPStage)
        assert process_nlp._vocab_hash_file == os.path.realpath(vocab_file_name)
        assert process_nlp._truncation
        assert process_nlp._do_lower_case
        assert not process_nlp._add_special_tokens

        assert isinstance(add_scores, AddScoresStage)
        assert isinstance(conv_msg, ConvMsg)
        assert isinstance(inf_ident, IdentityInferenceStage)

        assert isinstance(inf_pytorch, PyTorchInferenceStage)
        assert inf_pytorch._model_filename == tmp_model

        assert isinstance(mlflow_drift, MLFlowDriftStage)
        assert mlflow_drift._tracking_uri == mlflow_uri

        assert isinstance(triton_inf, TritonInferenceStage)
        assert triton_inf._model_name == 'test-model'
        assert triton_inf._server_url == 'test:123'
        assert triton_inf._force_convert_inputs

        assert isinstance(monitor, MonitorStage)
        assert monitor._mc._description == 'Unittest'
        assert monitor._mc._smoothing == 0.001
        assert monitor._mc._unit == 'inf'

        assert isinstance(add_class, AddClassificationsStage)
        assert add_class._labels == ('is_phishing', )
        assert add_class._threshold == 0.7

        assert isinstance(validation, ValidationStage)
        assert validation._results_file_name == 'results.json'
        assert validation._index_col == '_index_'

        # Click appears to be converting this into a tuple
        assert list(validation._exclude_columns) == ['event_dt']
        assert validation._rel_tol == 0.1

        assert isinstance(serialize, SerializeStage)
        assert isinstance(to_file, WriteToFileStage)
        assert to_file._controller._output_file == 'out.csv'

        assert isinstance(to_kafka, WriteToKafkaStage)
        assert to_kafka._kafka_conf['bootstrap.servers'] == 'kserv1:123,kserv2:321'
        assert to_kafka._output_topic == 'test_topic'

    @pytest.mark.replace_callback('pipeline_nlp')
    def test_pipeline_alias(self, config, callback_values):  # pylint: disable=unused-argument
        """
        Verify that pipeline implies pipeline-nlp
        """
        args = GENERAL_ARGS + ['pipeline'] + FILE_SRC_ARGS + TO_FILE_ARGS

        obj = {}
        runner = CliRunner()
        result = runner.invoke(commands.cli, args, obj=obj)
        assert result.exit_code == 47, result.output

        config = obj["config"]
        # Ensure our config is populated correctly
        assert config.mode == PipelineModes.NLP

    @pytest.mark.replace_callback('pipeline_nlp')
    def test_pipeline_nlp_relative_paths(self, config, callback_values, bert_cased_hash: str):
        """
        Ensure that the default paths in the nlp pipeline are valid when run from outside the morpheus repo
        """

        args = (GENERAL_ARGS + ['pipeline-nlp'] + FILE_SRC_ARGS + [
            'deserialize',
            'preprocess',
        ] + INF_TRITON_ARGS + MONITOR_ARGS + ['add-class'] + VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS)

        obj = {}
        runner = CliRunner()
        result = runner.invoke(commands.cli, args, obj=obj)
        assert result.exit_code == 47, result.output

        expected_labels = load_labels_file(os.path.join(TEST_DIRS.data_dir, 'labels_nlp.txt'))

        # Ensure our config is populated correctly
        config = obj["config"]
        assert config.class_labels == expected_labels

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        # pylint: disable=unused-variable
        [file_source, deserialize, process_nlp, triton_inf, monitor, add_class, validation, serialize, to_file] = stages

        assert process_nlp._vocab_hash_file == os.path.realpath(bert_cased_hash)

    @pytest.mark.replace_callback('pipeline_nlp')
    def test_pipeline_nlp_relative_path_precedence(self, config, callback_values, tmp_path):
        """
        Ensure that relative paths are choosen over the morpheus data directory paths
        """

        morpheus_root = os.path.dirname(morpheus.__file__)

        vocab_file = "data/bert-base-cased-hash.txt"
        labels_file = "data/labels_nlp.txt"

        vocab_file_local = os.path.join(tmp_path, vocab_file)
        labels_file_local = os.path.join(tmp_path, labels_file)

        shutil.copytree(os.path.join(morpheus_root, "data"), os.path.join(tmp_path, "data"), dirs_exist_ok=True)

        # Use different labels
        test_labels = ["label1", "label2", "label3"]

        # Overwrite the copied labels
        with open(labels_file_local, mode="w", encoding='UTF-8') as f:
            f.writelines("\n".join(test_labels))

        args = (GENERAL_ARGS + ['pipeline-nlp', f"--labels_file={labels_file}"] + FILE_SRC_ARGS +
                ['deserialize', 'preprocess', f"--vocab_hash_file={vocab_file}"] + INF_TRITON_ARGS + MONITOR_ARGS +
                ['add-class'] + VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS)

        obj = {}
        runner = CliRunner()
        result = runner.invoke(commands.cli, args, obj=obj)
        assert result.exit_code == 47, result.output

        # Ensure our config is populated correctly
        config = obj["config"]
        assert config.class_labels == test_labels

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        # pylint: disable=unused-variable
        [file_source, deserialize, process_nlp, triton_inf, monitor, add_class, validation, serialize, to_file] = stages

        assert process_nlp._vocab_hash_file == vocab_file_local

    # pylint: disable=unused-argument
    @pytest.mark.replace_callback('pipeline_fil')
    def test_pipeline_fil_relative_path_precedence(self, config: Config, tmp_path: str, callback_values: dict):
        """
        Ensure that relative paths are choosen over the morpheus data directory paths
        """

        labels_file = "data/labels_fil.txt"
        columns_file = "data/columns_fil.txt"

        labels_file_local = os.path.join(tmp_path, labels_file)
        columns_file_local = os.path.join(tmp_path, columns_file)

        os.makedirs(os.path.join(tmp_path, "data"), exist_ok=True)

        # Use different labels
        test_labels = ["label1"]

        # Overwrite the copied labels
        with open(labels_file_local, mode="w", encoding='UTF-8') as f:
            f.writelines("\n".join(test_labels))

        # Use different labels
        test_columns = [f"column{i}" for i in range(18)]

        # Overwrite the copied labels
        with open(columns_file_local, mode="w", encoding='UTF-8') as f:
            f.writelines("\n".join(test_columns))

        args = (GENERAL_ARGS + ['pipeline-fil', f"--labels_file={labels_file}", f"--columns_file={columns_file}"] +
                FILE_SRC_ARGS + ['deserialize', 'preprocess'] + INF_TRITON_ARGS + MONITOR_ARGS + ['add-class'] +
                VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS)

        obj = {}
        runner = CliRunner()
        result = runner.invoke(commands.cli, args, obj=obj)
        assert result.exit_code == 47, result.output

        # Ensure our config is populated correctly
        config = obj["config"]
        assert config.class_labels == test_labels

        assert config.fil.feature_columns == test_columns
