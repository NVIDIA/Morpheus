# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

import typing

from pymilvus import Collection
from pymilvus import MilvusClient as PyMilvusClient
from pymilvus.orm.mutation import MutationResult

DATA_TYPE_MAP = {
    "BOOL": 1,
    "INT8": 2,
    "INT16": 3,
    "INT32": 4,
    "INT64": 5,
    "FLOAT": 10,
    "DOUBLE": 11,
    "STRING": 20,
    "VARCHAR": 21,
    "ARRAY": 22,
    "JSON": 23,
    "BINARY_VECTOR": 100,
    "FLOAT_VECTOR": 101,
    "FLOAT16_VECTOR": 102,
    "BFLOAT16_VECTOR": 103,
    "UNKNOWN": 999
}


def handle_exceptions(func_name: str, error_message: str) -> typing.Callable:
    """
    Decorator function to handle exceptions and log errors.

    Parameters
    ----------
    func_name : str
        Name of the func being decorated.
    error_message : str
        Error message to log in case of an exception.

    Returns
    -------
    typing.Callable
        Decorated function.
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                raise RuntimeError(f"{error_message} - Failed to execute {func_name}") from ex

        return wrapper

    return decorator


class MilvusClient(PyMilvusClient):
    """
    Extension of the `MilvusClient` class with custom functions.

    Parameters
    ----------
    uri : str
        URI for connecting to Milvus server.
    user : str
        User name for authentication.
    password : str
        Password for authentication.
    db_name : str
        Name of the Milvus database.
    token : str
        Token for authentication.
    **kwargs : dict[str, typing.Any]
        Additional keyword arguments for the MilvusClient constructor.
    """

    def __init__(self, uri: str, user: str, password: str, db_name: str, token: str, **kwargs: dict[str, typing.Any]):
        super().__init__(uri=uri, user=user, password=password, db_name=db_name, token=token, **kwargs)

    @handle_exceptions("has_collection", "Error checking collection existence")
    def has_collection(self, collection_name: str) -> bool:
        """
        Check if a collection exists in the database.

        Parameters
        ----------
        collection_name : str
            Name of the collection to check.

        Returns
        -------
        bool
            True if the collection exists, False otherwise.
        """
        conn = self._get_connection()
        return conn.has_collection(collection_name)

    @handle_exceptions("create_partition", "Error creating partition")
    def create_partition(self, collection_name: str, partition_name: str, timeout: float = 1.0) -> None:
        """
        Create a partition within a collection.

        Parameters
        ----------
        collection_name : str
            Name of the collection.
        partition_name : str
            Name of the partition to create.
        timeout : float, optional
            Timeout for the operation in seconds (default is 1.0).
        """
        conn = self._get_connection()
        conn.create_partition(collection_name=collection_name, partition_name=partition_name, timeout=timeout)

    @handle_exceptions("load_collection", "Error loading collection")
    def load_collection(self, collection_name: str) -> None:
        """
        Load a collection into memory.

        Parameters
        ----------
        collection_name : str
            Name of the collection to load.
        """
        conn = self._get_connection()
        conn.load_collection(collection_name=collection_name)

    @handle_exceptions("release_collection", "Error releasing collection")
    def release_collection(self, collection_name: str) -> None:
        """
        Release a loaded collection from memory.

        Parameters
        ----------
        collection_name : str
            Name of the collection to release.
        """
        conn = self._get_connection()
        conn.release_collection(collection_name=collection_name)

    @handle_exceptions("upsert", "Error upserting collection entities")
    def upsert(self, collection_name: str, entities: list, **kwargs: dict[str, typing.Any]) -> MutationResult:
        """
        Upsert entities into a collection.

        Parameters
        ----------
        collection_name : str
            Name of the collection to upsert into.
        entities : list
            List of entities to upsert.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments for the upsert operation.

        Returns
        -------
        MutationResult
            Result of the upsert operation.
        """
        conn = self._get_connection()
        return conn.upsert(collection_name=collection_name, entities=entities, **kwargs)

    @handle_exceptions("delete_by_expr", "Error deleting collection entities")
    def delete_by_expr(self, collection_name: str, expression: str, **kwargs: dict[str, typing.Any]) -> MutationResult:
        """
        Delete entities from a collection using an expression.

        Parameters
        ----------
        collection_name : str
            Name of the collection to delete from.
        expression : str
            Deletion expression.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments for the delete operation.

        Returns
        -------
        MutationResult
            Returns result of delete operation.
        """
        conn = self._get_connection()
        return conn.delete(collection_name=collection_name, expression=expression, **kwargs)

    @handle_exceptions("has_partition", "Error checking partition existence")
    def has_partition(self, collection_name: str, partition_name: str) -> bool:
        """
        Check if a partition exists within a collection.

        Parameters
        ----------
        collection_name : str
            Name of the collection.
        partition_name : str
            Name of the partition to check.

        Returns
        -------
        bool
            True if the partition exists, False otherwise.
        """
        conn = self._get_connection()
        return conn.has_partition(collection_name=collection_name, partition_name=partition_name)

    @handle_exceptions("drop_partition", "Error dropping partition")
    def drop_partition(self, collection_name: str, partition_name: str) -> None:
        """
        Drop a partition from a collection.

        Parameters
        ----------
        collection_name : str
            Name of the collection.
        partition_name : str
            Name of the partition to drop.
        """
        conn = self._get_connection()
        conn.drop_partition(collection_name=collection_name, partition_name=partition_name)

    @handle_exceptions("drop_index", "Error dropping index")
    def drop_index(self, collection_name: str, field_name: str, index_name: str) -> None:
        """
        Drop an index from a collection.

        Parameters
        ----------
        collection_name : str
            Name of the collection.
        field_name : str
            Name of the field associated with the index.
        index_name : str
            Name of the index to drop.
        """
        conn = self._get_connection()
        conn.drop_index(collection_name=collection_name, field_name=field_name, index_name=index_name)

    @handle_exceptions("get_collection", "Error getting collection object")
    def get_collection(self, collection_name: str, **kwargs: dict[str, typing.Any]) -> Collection:
        """
        Returns `Collection` object associated with the given collection name.

        Parameters
        ----------
        collection_name : str
            Name of the collection to delete from.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments to get Collection instance.

        Returns
        -------
        Collection
            Returns pymilvus Collection instance.
        """
        collection = Collection(name=collection_name, using=self._using, **kwargs)

        return collection
