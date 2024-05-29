# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import logging
import pickle
import typing

import mrc

from morpheus.config import Config
from morpheus.messages import ControlMessage
from morpheus.messages import MultiResponseMessage
from morpheus.messages.multi_message import MultiMessage
from morpheus.modules.output.write_to_vector_db import WriteToVectorDBLoaderFactory
from morpheus.pipeline.pass_thru_type_mixin import PassThruTypeMixin
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.service.vdb.vector_db_service import VectorDbServiceProvider
from morpheus.service.vdb.utils import VectorDBServiceProviderFactory
from morpheus.utils.module_utils import ModuleLoader

logger = logging.getLogger(__name__)


class WriteToVectorDBStage(PassThruTypeMixin, SinglePortStage):
    """
    Writes messages to a Vector Database.

    Parameters
    ----------
    config : Config
        Pipeline configuration instance.
    service : typing.Union[str, VectorDBService]
        Either the name of the vector database service to use or an instance of VectorDBService
        for managing the resource.
    resource_name : str
        The identifier of the resource on which operations are to be performed in the vector database.
    embedding_column_name : str, optional
        Name of the embedding column, by default "embedding".
    recreate : bool, optional
        Specifies whether to recreate the resource if it already exists, by default False.
    resource_kwargs : dict, optional
        Additional keyword arguments to pass when performing vector database writes on a given resource.
    batch_size : int
        Accumulates messages until reaching the specified batch size for writing to VDB.
    write_time_interval : float
        Specifies the time interval (in seconds) for writing messages, or writing messages
        when the accumulated batch size is reached.

    Raises
    ------
    ValueError
        If `service` is not a valid string (service name) or an instance of VectorDbServiceProvider.
    """

    def __init__(self,
                 config: Config,
                 service: typing.Union[str, VectorDbServiceProvider],
                 resource_name: str,
                 embedding_column_name: str = "embedding",
                 recreate: bool = False,
                 resource_kwargs: dict = None,
                 batch_size: int = 1024,
                 write_time_interval: float = 3.0,
                 resource_schemas: dict = None,
                 **service_kwargs):

        super().__init__(config)

        resource_kwargs = resource_kwargs if resource_kwargs is not None else {}
        resource_schemas = resource_schemas if resource_schemas is not None else {}

        if isinstance(service, str):
            service = VectorDBServiceProviderFactory.create_instance(service, **service_kwargs)

        module_config = {
            "batch_size": batch_size,
            "default_resource_name": resource_name,
            "embedding_column_name": embedding_column_name,
            "recreate": recreate,
            "resource_kwargs": resource_kwargs,
            "resource_schemas": resource_schemas,
            "service_provider": str(pickle.dumps(service), encoding="latin1"),
            "write_time_interval": write_time_interval
        }

        module_name = f"write_to_vector_db__{resource_name}"

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Module will be loading with name: %s", module_name)

        self._module_loader: ModuleLoader = WriteToVectorDBLoaderFactory.get_instance(module_name, module_config)

    @property
    def name(self) -> str:
        return "to-vector-db"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple(ControlMessage, MultiResponseMessage, MultiMessage)
            Accepted input types.

        """
        return (ControlMessage, MultiResponseMessage, MultiMessage)

    def supports_cpp_node(self):
        """Indicates whether this stage supports a C++ node."""
        return False

    def _build_single(self, builder: mrc.Builder, input_node: mrc.SegmentObject) -> mrc.SegmentObject:

        module = self._module_loader.load(builder)

        # Input and Output port names should be same as input and output port names of write_to_vector_db module.
        mod_in_node = module.input_port("input")
        mod_out_node = module.output_port("output")

        builder.make_edge(input_node, mod_in_node)

        return mod_out_node
