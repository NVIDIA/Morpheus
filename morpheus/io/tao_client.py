# Copyright (c) 2023, NVIDIA CORPORATION.
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

import functools
import io
import json
import logging
import os
import tarfile
import time
import typing

import requests

logger = logging.getLogger(__name__)

_KIND = typing.Literal["model", "dataset"]
_DATASET_ACTIONS = typing.Literal["convert", "convert_index", "convert_efficientdet"]
_MODEL_ACTIONS = typing.Literal["train", "evaluate", "prune", "retrain", "export", "inference"]


def validate_kind(func: typing.Callable) -> typing.Callable:
    """
    Validates given endpoint category.

    Parameters
    ----------
    func : typing.Callable
        Function that requires wrapping.
    Returns
    -------
    inner_func : typing.Callable
        Encapsulated function.
    """

    @functools.wraps(func)
    def inner_func(*args, **kwargs):
        if len(args) < 2:
            raise ValueError("Kind not found. Select from available kinds: {}".format(_KIND))
        kind = args[1]

        if kind is None:
            raise TypeError("TypeError: a string-like object is required for kind, not 'NoneType'")
        if kind not in typing.get_args(_KIND):
            raise ValueError("Invalid kind '{}'. Available kinds are {}".format(kind, _KIND))
        return func(*args, **kwargs)

    return inner_func


def validate_actions(func: typing.Callable) -> typing.Callable:
    """
    Validates TAO actions.

    Parameters
    ----------
    func : typing.Callable
        Function that requires wrapping.
    Returns
    -------
    inner_func : typing.Callable
        Encapsulated function.
    """

    @functools.wraps(func)
    def inner_func(*args, **kwargs):

        actions_by_kind = _DATASET_ACTIONS
        if args[1] == "model":
            actions_by_kind = _MODEL_ACTIONS

        if len(args) < 3:
            raise ValueError("Actions not found. Select from available actions: {}".format(actions_by_kind))

        actions = args[2]

        if actions is None:
            raise TypeError("TypeError: a string-like object is required for an action, not 'NoneType'")

        available_actions = typing.get_args(actions_by_kind)

        if isinstance(actions, list):
            if not set(actions).issubset(available_actions):
                raise ValueError("One or more actions are not valid actions '{}'. Available actions are {}".format(
                    actions, actions_by_kind))
        else:
            if actions not in available_actions:
                raise ValueError("Invalid action '{}'. Available actions are {}".format(actions, actions_by_kind))

        return func(*args, **kwargs)

    return inner_func


def generate_schema_url(url: str, ssl: bool) -> str:
    """
    Generates url with schema.

    Parameters
    ----------
    url : str
        URL
    ssl : str
        Determines whether to use HTTP or HTTPS in the schema.
    Returns
    -------
    url : str
        Schema attached url.
    """

    if url.startswith("http://") or url.startswith("https://"):
        raise ValueError("URL should not include the scheme")

    scheme = "https://" if ssl else "http://"
    url = scheme + (url if url[-1] != "/" else url[:-1])

    return url


def apikey_type_check(apikey: str):
    """
    Verify API key type.

    Parameters
    ----------
    apikey : str
        NGC API key
    """

    if not isinstance(apikey, str):
        raise ValueError('API key must be a string')

    if not apikey:
        raise ValueError('API key can not be an empty string')


class TaoApiClient():
    """
    This serves as a client wrapper for TAO REST endpoints. This class gives you the flexibility to upload and delete
    datasets, run training or inference processes, and update configurations on the TAO toolkit API server.

    Parameters
    ----------
    apikey : str
        NGC API key.
    url : str
        TAO toolkit API URL.
    ssl : str
        Determines whether to use HTTP or HTTPS in the schema.
    cert : str
        Client side certificate.
    server_side_cert : str
        Server side certificate.
    proxies : typing.Dict[str, str]
        Defines the HTTP and HTTPS connections.
    """

    def __init__(self,
                 apikey: str,
                 url: str,
                 ssl: bool = False,
                 cert: str = None,
                 server_side_cert: bool = True,
                 proxies: typing.Dict[str, str] = None):

        apikey_type_check(apikey)
        self._apikey = apikey
        self._parsed_url = generate_schema_url(url, ssl)
        self._base_uri = f"{self._parsed_url}/api/v1"
        self._ssl = ssl
        self._user_uri = None

        self._session = requests.Session()

        if server_side_cert:
            self._session.verify = cert
        self._session.cert = cert

        if proxies:
            self._session.proxies.update(proxies)

    def authorize(self):
        """
        This function establishes a session and authorizes with the TAO toolkit API server.
        """

        endpoint = f"{self._base_uri}/login/{self._apikey}"

        logger.debug("Login endpoint: {}".format(endpoint))

        resp = self.session.get(endpoint)
        if not resp.status_code == 200:
            raise Exception("Login failed: {}".format(resp.content))

        logger.info("Login has been successful!")

        json_resp = resp.json()

        self._user_uri = self._base_uri + "/user/" + json_resp.get("user_id")

        if not self._ssl:
            self._session.headers.update({'Authorization': 'Bearer ' + json_resp.get("token")})

    @property
    def base_uri(self) -> str:
        return self._base_uri

    @property
    def user_uri(self) -> str:
        return self._user_uri

    @property
    def session(self) -> requests.Session:
        return self._session

    @validate_kind
    def create_resource(self, kind: _KIND, data: typing.Dict, **kwargs) -> str:
        """
        Create new resource.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        data : typing.Dict
            Initial metadata for new resource.
        **kwargs :
            Additional arguments.
        Returns
        -------
        resource_id: str
            Unique identifier for the resource.
        """

        data = json.dumps(data)

        endpoint = f"{self.user_uri}/{kind}"

        logger.debug("create resource with endpoint: {}".format(endpoint))

        resp = self.session.post(endpoint, data=data, **kwargs)
        if not resp.status_code == 201:
            raise Exception("Error creating resource {} with endpoint {}: {}".format(kind, endpoint, resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        resource_id = json_resp.get("id")

        return resource_id

    @validate_kind
    def partial_update_resource(self, kind: _KIND, data: typing.Dict, resource_id: str, **kwargs) -> typing.Dict:
        """
        Partially update the resource.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        data : typing.Dict
            Metadata that needs to be updated.
        resource_id: str
            Unique identifier for the resource.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        data = json.dumps(data)

        endpoint = f"{self.user_uri}/{kind}/{resource_id}"
        logger.debug("Partially update resource with endpoint: {}".format(endpoint))

        resp = self.session.patch(endpoint, data=data, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Unable to partially update resource: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    @validate_kind
    def update_resource(self, kind: _KIND, data: typing.Dict, resource_id: str, **kwargs) -> typing.Dict:
        """
        Update the resource.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        data : typing.Dict
            Metadata that needs to be updated.
        resource_id: str
            Unique identifier for the resource.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        data = json.dumps(data)

        endpoint = f"{self.user_uri}/{kind}/{resource_id}"
        logger.debug("Update resource with endpoint: {}".format(endpoint))

        resp = self.session.put(endpoint, data=data, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Unable to update resource: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    @validate_kind
    def upload_resource(self, kind: _KIND, resource_path: str, resource_id: str, **kwargs) -> typing.Dict:
        """
        Upload the resource.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        resource_path : str
            The location of the resource to be uploaded.
        resource_id: str
            Unique identifier for the resource.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        if os.path.exists(resource_path):
            if os.path.isfile(resource_path):
                files = [("file", open(resource_path, "rb"))]
            else:
                raise Exception("Resource path must be a file.")
        else:
            raise ValueError("Resource path provided does not exists.")

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/upload"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.post(endpoint, files=files, **kwargs)
        if not resp.status_code == 201:
            raise Exception("Unable to upload resource: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def upload_in_memory_data(self, files_content: typing.Dict[str, bytearray], resource_id: str,
                              **kwargs) -> typing.Dict:
        """
        Upload in memory data.

        Parameters
        ----------
        files_content : typing.Dict[str, bytearray]
            Keys in the dictionary are file path and values are file content as bytearray.
        resource_id: str
            Unique identifier for the resource.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """
        # Create an in-memory binary stream
        tar_bytes = io.BytesIO()

        # Open tarfile
        tar = tarfile.open(fileobj=tar_bytes, mode='w')

        # Create file within the directories.
        for key in files_content.keys():
            # Here key is a filepath.
            file_content = files_content[key]
            file_content = io.BytesIO(file_content)
            file_info = tarfile.TarInfo(name=key)
            file_info.size = len(file_content.getvalue())
            file_info.mtime = time.time()
            tar.addfile(file_info, file_content)

        # Seek to the beginning of the stream
        tar_bytes.seek(0)

        endpoint = f"{self.user_uri}/dataset/{resource_id}/upload"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.post(endpoint, files={'file': tar_bytes}, **kwargs)

        # Close tar file.
        tar.close()

        if not resp.status_code == 201:
            raise Exception("Unable to upload resource: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    @validate_kind
    def list_resources(self, kind: _KIND, **kwargs) -> typing.Dict:
        """
        List available resources by kind.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.get(endpoint, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Unable to list the resources: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    @validate_kind
    @validate_actions
    def get_specs_schema(self, kind: _KIND, action: str, resource_id: str, **kwargs) -> typing.Dict:
        """
        Get specs schema by kind and action.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        action: str
            TAO actions.
        resource_id: str
            Unique identifier for the resource.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/specs/{action}/schema"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.get(endpoint, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Error getting specs schema: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    @validate_kind
    @validate_actions
    def get_specs(self, kind: _KIND, action: str, resource_id: str, **kwargs) -> typing.Dict:
        """
        Get specs by kind and action.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        action: str
            TAO actions.
        resource_id: str
            Unique identifier for the resource.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/specs/{action}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.get(endpoint, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Error getting specs: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    @validate_kind
    @validate_actions
    def update_specs(self, kind: _KIND, action: str, specs: typing.Dict, resource_id: str, **kwargs) -> typing.Dict:
        """
        Update specs by kind and action.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        action: str
            TAO actions.
        specs: typing.Dict
            Updated specs.
        resource_id: str
            Unique identifier for the resource.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/specs/{action}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        data = json.dumps(specs)

        resp = self.session.post(endpoint, data=data, **kwargs)
        if not resp.status_code == 201:
            raise Exception("Unable to update specs: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    @validate_kind
    @validate_actions
    def save_specs(self, kind: _KIND, action: str, resource_id: str, **kwargs) -> typing.Dict:
        """
        Save specs by kind and action.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        action: str
            TAO actions.
        resource_id: str
            Unique identifier for the resource.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/specs/{action}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.post(endpoint, **kwargs)
        if not resp.status_code == 201:
            raise Exception("Error saving specs: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    @validate_kind
    @validate_actions
    def run_job(self,
                kind: _KIND,
                actions: typing.List[str],
                resource_id: str,
                parent_job_id: str = None,
                **kwargs) -> typing.Dict:
        """
        Run job by kind and action.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        action: str
            TAO actions.
        resource_id: str
            Unique identifier for the resource.
        parent_job_id: str
            Parent job id.
        **kwargs :
            Additional arguments.
        Returns
        -------
        job_ids : typing.List[str]
            List of job id's by actions.
        """

        data = json.dumps({"job": parent_job_id, "actions": actions})

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/job"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.post(endpoint, data=data, **kwargs)
        if not resp.status_code == 201:
            raise Exception("Unable to run the job: {}".format(resp.content))

        job_ids = resp.json()
        logger.debug("Response: {}".format(job_ids))

        return job_ids

    @validate_kind
    def get_job_status(self, kind: _KIND, resource_id: str, job_id: str, **kwargs) -> typing.Dict:
        """
        Get job status.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        resource_id: str
            Unique identifier for the resource.
        job_id: str
            Unique identifier for the job.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/job/{job_id}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.get(endpoint, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Unable to retrieve job status: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    @validate_kind
    def list_jobs(self, kind: _KIND, resource_id: str, **kwargs) -> typing.Dict:
        """
        List jobs for a given resource by kind.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        resource_id: str
            Unique identifier for the resource.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/job"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.get(endpoint, **kwargs)

        if not resp.status_code == 200:
            raise Exception("Error retrieving list of jobs belongs to {}: {}".format(kind, resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    @validate_kind
    def delete_job(self, kind: _KIND, resource_id: str, job_id: str, **kwargs) -> typing.Dict:
        """
        Delete job for a given kind and resource identifier.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        resource_id: str
            Unique identifier for the resource.
        job_id: str
            Unique identifier for the job.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/job/{job_id}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.delete(endpoint, **kwargs)
        if not resp.status_code == 200:
            raise Exception("Unable to delete job belongs to {} group: {}".format(kind, resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    @validate_kind
    def cancel_job(self, kind: _KIND, resource_id: str, job_id: str, **kwargs) -> typing.Dict:
        """
        Cancel job for a given kind and resource identifier.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        resource_id: str
            Unique identifier for the resource.
        job_id: str
            Unique identifier for the job.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/{kind}/{resource_id}/job/{job_id}/cancel"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.post(endpoint, **kwargs)

        if not resp.status_code == 200:
            raise Exception("Unable to cancel {} job: {}".format(kind, resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def resume_model_job(self, model_id: str, job_id: str, **kwargs) -> typing.Dict:
        """
        Resume model job.

        Parameters
        ----------
        model_id: str
            Unique identifier for the model.
        job_id: str
            Unique identifier for the job.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """

        endpoint = f"{self.user_uri}/model/{model_id}/job/{job_id}/resume"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.post(endpoint, **kwargs)

        if not resp.status_code == 200:
            raise Exception("Error resuming model job: {}".format(resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    @validate_kind
    def download_resource(self, kind: _KIND, resource_id, job_id, output_dir: str, **kwargs) -> str:
        """
        Download resources.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        resource_id: str
            Unique identifier for the resource.
        job_id: str
            Unique identifier for the job.
        output_dir : str
            Output directory to save the downloaded content.
        **kwargs :
            Additional arguments.
        Returns
        -------
        downloaded_path : str
            The download location's path.
        """
        job_status = self.get_job_status(kind, resource_id=resource_id, job_id=job_id)

        status = job_status.get("status")

        if status == "Done":

            endpoint = f'{self.user_uri}/{kind}/{resource_id}/job/{job_id}/download'
            logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

            resp = self.session.get(endpoint, **kwargs)

            if not resp.status_code == 200:
                raise Exception("Error downloading the job content: {}".format(resp.content))

            temp_tar = f'{job_id}.tar.gz'

            with open(temp_tar, 'wb') as f:
                f.write(resp.content)
            logger.debug("Untarring {}...".format(temp_tar))
            tar_command = f"tar -xvf {temp_tar} -C {output_dir}/"
            os.system(tar_command)
            logger.debug("Untarring {}... Done".format(temp_tar))
            os.remove(temp_tar)
            downloaded_path = f"{output_dir}/{job_id}"

            logger.debug("Results at location {}".format(downloaded_path))

            return downloaded_path

        logger.info("Resource can be downloaded only when the job is completed. Current status is in {}".format(status))

    @validate_kind
    def delete_resource(self, kind: _KIND, resource_id: str, **kwargs) -> typing.Dict:
        """
        Delete resource.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        resource_id: str
            Unique identifier for the resource.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """
        endpoint = f"{self.user_uri}/{kind}/{resource_id}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.delete(endpoint, **kwargs)

        if not resp.status_code == 200:
            raise Exception("Error deleting resource from {}: {}".format(kind, resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    @validate_kind
    def retrieve_resource(self, kind: _KIND, resource_id: str, **kwargs) -> typing.Dict:
        """
        Retrieve resource metadata.

        Parameters
        ----------
        kind : _KIND
            Endpoint type that is specific to a model or a dataset.
        resource_id: str
            Unique identifier for the resource.
        **kwargs :
            Additional arguments.
        Returns
        -------
        json_resp : typing.Dict
            JSON response.
        """
        endpoint = f"{self.user_uri}/{kind}/{resource_id}"
        logger.debug("Constructed endpoint with provided input: {}".format(endpoint))

        resp = self.session.get(endpoint, **kwargs)

        if not resp.status_code == 200:
            raise Exception("Error retrieving resource from {}: {}".format(kind, resp.content))

        json_resp = resp.json()
        logger.debug("Response: {}".format(json_resp))

        return json_resp

    def close(self):
        """
        Closes session.
        """
        session = getattr(self, '_session', None)
        if session:
            logger.debug("Closing session...")
            session.close()
            self._session = None
            logger.debug("Closing session... Done")
