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

import hashlib
import logging
import os
import typing
import urllib.parse

import mlflow
import mlflow.types
import pandas as pd
import requests
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.protos.databricks_pb2 import ErrorCode
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking import MlflowClient
from mlflow.types import ColSpec
from mlflow.types import Schema
from mlflow.types.utils import _infer_pandas_column as _mlflow_infer_pandas_column
from mlflow.types.utils import _infer_schema

from morpheus.messages import ControlMessage
from morpheus.utils.type_utils import is_cudf_type

if typing.TYPE_CHECKING:
    from morpheus.models.dfencoder import AutoEncoder

logger = logging.getLogger(__name__)


def _infer_pandas_column(col: pd.Series):
    try:
        return _mlflow_infer_pandas_column(col)
    except MlflowException as ex:
        # mlflow internally calls `pd.api.types.is_string_dtype(col)` which was changed in Pandas 2.
        # passing the series such as `[None]` will return False when the dtype is string, so
        # it is necessary to pass the dtype of the series rather than the series itself for backwards
        # compatibility.
        if "Failed to infer schema for pandas.Series" in ex.message and pd.api.types.is_string_dtype(col.dtype):
            return mlflow.types.DataType.string
        raise ex


class MLFlowModelWriterController:
    """
    Controller class for writing machine learning models to MLflow with optional permissions and configurations.

    Parameters
    ----------
    model_name_formatter : str
        Model name formatter.
    experiment_name_formatter : str
        Experiment name formatter.
    databricks_permissions : dict
        Users with read/write permissions.
    conda_env : dict
        Conda environment.
    timeout :
        Timeout for get requests.
    timestamp_column_name :
        Timestamp column name to be used from the dataframe.

    """

    def __init__(self,
                 model_name_formatter,
                 experiment_name_formatter,
                 databricks_permissions,
                 conda_env,
                 timeout,
                 timestamp_column_name):
        self._model_name_formatter = model_name_formatter
        self._experiment_name_formatter = experiment_name_formatter
        self._databricks_permissions = databricks_permissions
        self._conda_env = conda_env
        self._timeout = timeout
        self._timestamp_column_name = timestamp_column_name

    @property
    def model_name_formatter(self):
        return self._model_name_formatter

    @property
    def experiment_name_formatter(self):
        return self._experiment_name_formatter

    @property
    def databricks_permissions(self):
        return self._databricks_permissions

    def _create_safe_user_id(self, user_id: str):
        """
        Creates a safe user ID for use in MLflow model names and experiment names.

        Parameters
        ----------
        user_id : str
            The user ID.

        Returns
        -------
        str
            The generated safe user ID.
        """

        safe_user_id = user_id.replace('.', '_dot_')
        safe_user_id = safe_user_id.replace('/', '_slash_')
        safe_user_id = safe_user_id.replace(':', '_colon_')

        return safe_user_id

    def user_id_to_model(self, user_id: str):
        """
        Converts a user ID to an model name

        Parameters
        ----------
        user_id : str
            The user ID.

        Returns
        -------
        str
            The generated model name.
        """

        kwargs = {
            "user_id": self._create_safe_user_id(user_id),
            "user_md5": hashlib.md5(user_id.encode('utf-8')).hexdigest(),
        }

        return self._model_name_formatter.format(**kwargs)

    def user_id_to_experiment(self, user_id: str) -> str:
        """
        Converts a user ID to an experiment name

        Parameters
        ----------
        user_id : str
            The user ID.

        Returns
        -------
        str
            The generated experiment name.
        """

        safe_user_id = self._create_safe_user_id(user_id)

        kwargs = {
            "user_id": safe_user_id,
            "user_md5": hashlib.md5(safe_user_id.encode('utf-8')).hexdigest(),
            "reg_model_name": self.user_id_to_model(user_id=user_id)
        }

        return self._experiment_name_formatter.format(**kwargs)

    def _apply_model_permissions(self, reg_model_name: str):

        # Check the required variables
        databricks_host = os.environ.get("DATABRICKS_HOST", None)
        databricks_token = os.environ.get("DATABRICKS_TOKEN", None)

        if (databricks_host is None or databricks_token is None):
            raise RuntimeError("Cannot set Databricks model permissions. "
                               "Environment variables `DATABRICKS_HOST` and `DATABRICKS_TOKEN` must be set")

        headers = {"Authorization": f"Bearer {databricks_token}"}

        url_base = f"{databricks_host}"

        try:
            # First get the registered model ID
            get_registered_model_url = urllib.parse.urljoin(url_base,
                                                            "/api/2.0/mlflow/databricks/registered-models/get")

            get_registered_model_response = requests.get(url=get_registered_model_url,
                                                         headers=headers,
                                                         params={"name": reg_model_name},
                                                         timeout=self._timeout)

            registered_model_response = get_registered_model_response.json()

            reg_model_id = registered_model_response["registered_model_databricks"]["id"]

            # Now apply the permissions. If it exists already, it will be overwritten or it is a no-op
            patch_registered_model_permissions_url = urllib.parse.urljoin(
                url_base, f"/api/2.0/preview/permissions/registered-models/{reg_model_id}")

            patch_registered_model_permissions_body = {
                "access_control_list": [{
                    "group_name": group, "permission_level": permission
                } for group,
                                        permission in self._databricks_permissions.items()]
            }

            requests.patch(url=patch_registered_model_permissions_url,
                           headers=headers,
                           json=patch_registered_model_permissions_body,
                           timeout=self._timeout)

        except Exception:
            logger.exception("Error occurred trying to apply model permissions to model: %s",
                             reg_model_name,
                             exc_info=True)

    def on_data(self, message: ControlMessage) -> ControlMessage:
        """
        Stores incoming models into MLflow.

        Parameters
        ----------
        message : ControlMessage
            The incoming message containing the model and related metadata.

        Returns
        -------
        ControlMessage
            The processed message.
        """

        user = message.get_metadata("user_id")

        model: "AutoEncoder" = message.get_metadata("model")

        model_path = "dfencoder"
        reg_model_name = self.user_id_to_model(user_id=user)

        # Write to ML Flow
        try:
            mlflow.end_run()

            experiment_name = self.user_id_to_experiment(user_id=user)

            # Creates a new experiment if it doesn't exist
            experiment = mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="autoencoder model training run",
                                  experiment_id=experiment.experiment_id) as run:

                model_path = f"{model_path}-{run.info.run_uuid}"

                # Log all params in one dict to avoid round trips
                mlflow.log_params({
                    "Algorithm": "Denosing Autoencoder",
                    "Epochs": model.learning_rate_decay.state_dict().get("last_epoch", "unknown"),
                    "Learning rate": model.learning_rate,
                    "Batch size": model.batch_size,
                    "Start Epoch": message.payload().get_data(self._timestamp_column_name).min(),
                    "End Epoch": message.payload().get_data(self._timestamp_column_name).max(),
                    "Log Count": message.payload().count,
                })

                metrics_dict: typing.Dict[str, float] = {}

                # Add info on the embeddings
                for key, value in model.categorical_fts.items():
                    embedding = value.get("embedding", None)

                    if (embedding is None):
                        continue

                    metrics_dict[f"embedding-{key}-num_embeddings"] = embedding.num_embeddings
                    metrics_dict[f"embedding-{key}-embedding_dim"] = embedding.embedding_dim

                mlflow.log_metrics(metrics_dict)

                # Use the prepare_df function to setup the direct inputs to the model. Only include features returned by
                # prepare_df to show the actual inputs to the model (any extra are discarded)
                input_df = message.payload().get_data().iloc[0:1]

                if is_cudf_type(input_df):
                    input_df = input_df.to_pandas()

                prepared_df = model.prepare_df(input_df)
                output_values = model.get_anomaly_score(input_df)

                input_schema = Schema([
                    ColSpec(type=_infer_pandas_column(input_df[col_name]), name=col_name)
                    for col_name in list(prepared_df.columns)
                ])
                output_schema = _infer_schema(output_values)

                model_sig = ModelSignature(inputs=input_schema, outputs=output_schema)

                model_info = mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=model_path,
                    conda_env=self._conda_env,
                    signature=model_sig,
                )

                client = MlflowClient()

                # First ensure a registered model has been created
                try:
                    create_model_response = client.create_registered_model(reg_model_name)
                    logger.debug("Successfully registered model '%s'.", create_model_response.name)
                except MlflowException as e:
                    if e.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                        pass
                    else:
                        raise e

                # If we are using databricks, make sure we set the correct permissions
                if (self._databricks_permissions is not None and mlflow.get_tracking_uri() == "databricks"):
                    # Need to apply permissions
                    self._apply_model_permissions(reg_model_name=reg_model_name)

                model_src = RunsArtifactRepository.get_underlying_uri(model_info.model_uri)

                tags = {
                    "start": message.payload().get_data(self._timestamp_column_name).min(),
                    "end": message.payload().get_data(self._timestamp_column_name).max(),
                    "count": message.payload().get_data(self._timestamp_column_name).count()
                }

                # Now create the model version
                mv_obj = client.create_model_version(name=reg_model_name,
                                                     source=model_src,
                                                     run_id=run.info.run_id,
                                                     tags=tags)

                logger.debug("ML Flow model upload complete: %s:%s:%s", user, reg_model_name, mv_obj.version)

        except Exception:
            logger.exception("Error uploading model to ML Flow", exc_info=True)

        return message
