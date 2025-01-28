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

import copy
import json
import logging
import threading
import time
import typing
from functools import wraps

from morpheus.io.utils import cudf_string_cols_exceed_max_bytes
from morpheus.io.utils import truncate_string_cols_by_bytes
from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_utils import is_cudf_type
from morpheus_llm.error import IMPORT_ERROR_MESSAGE
from morpheus_llm.service.vdb.vector_db_service import VectorDBResourceService
from morpheus_llm.service.vdb.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None

# Milvus has a max string length in bytes of 65,535. Multi-byte characters like "ñ" will have a string length of 1, the
# byte length encoded as UTF-8 will be 2
# https://milvus.io/docs/limitations.md#Length-of-a-string
MAX_STRING_LENGTH_BYTES = 65_535

try:
    import pymilvus
    from pymilvus.orm.mutation import MutationResult

    from morpheus_llm.service.vdb.milvus_client import MilvusClient  # pylint: disable=ungrouped-imports
except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc


class FieldSchemaEncoder(json.JSONEncoder):

    def default(self, o: typing.Any) -> str:
        """
        Serialize objects to a JSON-compatible string format.

        Parameters
        ----------
        o : typing.Any
            Object to be serialized.

        Returns
        -------
        str
            JSON-compatible string format of the object.
        """

        if isinstance(o, pymilvus.DataType):
            return str(o)
        return json.JSONEncoder.default(self, o)

    @staticmethod
    def object_hook(obj: dict) -> dict:
        """
        Updated dictionary with pymilvus datatype.

        Parameters
        ----------
        obj : dict
            Dictionary to be converted.

        Returns
        -------
        dict
            Dictionary with changes to its original format.
        """

        if "type" in obj and "DataType." in obj["type"]:
            obj["type"] = getattr(pymilvus.DataType, obj["type"].split(".")[1])
        return obj

    @staticmethod
    def dump(field: "pymilvus.FieldSchema", f: typing.IO) -> str:
        """
        Serialize a FieldSchema object to a JSON file.

        Parameters
        ----------
        field : pymilvus.FieldSchema
            FieldSchema object to be serialized.
        f : typing.IO
            File-like object to which the data is serialized.

        Returns
        -------
        str
            JSON string.
        """
        return json.dump(field, f, cls=FieldSchemaEncoder)

    @staticmethod
    def dumps(field: "pymilvus.FieldSchema") -> str:
        """
        Serialize a FieldSchema object to a JSON-compatible string format.

        Parameters
        ----------
        field : pymilvus.FieldSchema
            FieldSchema object to be serialized.

        Returns
        -------
        str
            JSON-compatible string format of the FieldSchema object.
        """

        return json.dumps(field, cls=FieldSchemaEncoder)

    @staticmethod
    def load(f_obj: typing.IO) -> "pymilvus.FieldSchema":
        """
        Deserialize a JSON file to a FieldSchema object.

        Parameters
        ----------
        f_obj : typing.IO
            File-like object from which the data is deserialized.

        Returns
        -------
        pymilvus.FieldSchema
            Deserialized FieldSchema object.
        """
        return pymilvus.FieldSchema.construct_from_dict(json.load(f_obj, object_hook=FieldSchemaEncoder.object_hook))

    @staticmethod
    def loads(field: str) -> "pymilvus.FieldSchema":
        """
        Deserialize a JSON-compatible string to a FieldSchema object.

        Parameters
        ----------
        field : str
            JSON-compatible string to be deserialized.

        Returns
        -------
        pymilvus.FieldSchema
            Deserialized FieldSchema object.
        """

        return pymilvus.FieldSchema.construct_from_dict(json.loads(field, object_hook=FieldSchemaEncoder.object_hook))

    @staticmethod
    def from_dict(field: dict) -> "pymilvus.FieldSchema":
        """
        Convert a dictionary to a FieldSchema object.

        Parameters
        ----------
        field : dict
            Dictionary to be converted to a FieldSchema object.

        Returns
        -------
        pymilvus.FieldSchema
            Converted FieldSchema object.
        """

        # FieldSchema converts dtype -> type when serialized. We need to convert any dtype to type before deserilaizing

        # First convert any dtype to type
        if ("dtype" in field):
            field["type"] = field["dtype"]
            del field["dtype"]

        # Convert string type to DataType
        if ("type" in field and isinstance(field["type"], str)):
            field = FieldSchemaEncoder.object_hook(field)

        # Now use the normal from dict function
        return pymilvus.FieldSchema.construct_from_dict(field)


def with_collection_lock(func: typing.Callable) -> typing.Callable:
    """
    A decorator to synchronize access to a collection with a lock. This decorator ensures that operations on a
    specific collection within the Milvus Vector Database are synchronized by acquiring and
    releasing a collection-specific lock.

    Parameters
    ----------
    func : Callable
        The function to be wrapped with the lock.

    Returns
    -------
    Callable
        The wrapped function with the lock acquisition logic.
    """

    @wraps(func)
    def wrapper(self, name, *args, **kwargs):
        collection_lock = MilvusVectorDBService.get_collection_lock(name)
        with collection_lock:
            result = func(self, name, *args, **kwargs)
            return result

    return wrapper


class MilvusVectorDBResourceService(VectorDBResourceService):
    """
    Represents a service for managing resources in a Milvus Vector Database.

    Parameters
    ----------
    name : str
        Name of the resource.
    client : MilvusClient
        An instance of the MilvusClient for interaction with the Milvus Vector Database.
    truncate_long_strings : bool, optional
        When true, truncate strings values that are longer than the max length of the field
    """

    def __init__(self, name: str, client: "MilvusClient", truncate_long_strings: bool = False) -> None:
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE.format(package='pymilvus')) from IMPORT_EXCEPTION

        super().__init__()

        self._name = name
        self._client = client

        self._collection = self._client.get_collection(collection_name=self._name)
        self._fields: list[pymilvus.FieldSchema] = self._collection.schema.fields

        self._vector_field = None
        self._fillna_fields_dict = {}

        # Mapping of field name to max length for string fields
        self._fields_max_length: dict[str, int] = {}

        for field in self._fields:
            if field.dtype == pymilvus.DataType.FLOAT_VECTOR:
                self._vector_field = field.name
            else:
                # Intentionally excluding pymilvus.DataType.STRING, in our current version it isn't supported, and in
                # some database systems string types don't have a max length.
                if field.dtype == pymilvus.DataType.VARCHAR:
                    max_length = field.params.get('max_length')
                    if max_length is not None:
                        self._fields_max_length[field.name] = max_length
                if not field.auto_id:
                    self._fillna_fields_dict[field.name] = field.dtype

        self._truncate_long_strings = truncate_long_strings

        self._collection.load()
        print("Resource created ...")

    def _set_up_collection(self):
        """
        Set up the collection fields.
        """
        self._fields = self._collection.schema.fields

    def insert(self, data: list[list] | list[dict], **kwargs: dict[str, typing.Any]) -> dict:
        """
        Insert data into the vector database.

        Parameters
        ----------
        data : list[list] | list[dict]
            Data to be inserted into the collection.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns response content as a dictionary.
        """
        result = self._collection.insert(data, **kwargs)
        self._collection.flush()

        return self._insert_result_to_dict(result=result)

    def insert_dataframe(self, df: DataFrameType, **kwargs: dict[str, typing.Any]) -> dict:
        """
        Insert a dataframe entires into the vector database.

        Parameters
        ----------
        df : DataFrameType
            Dataframe to be inserted into the collection.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns response content as a dictionary.
        """
        # Ensure that there are no None values in the DataFrame entries.
        print("##### 1")
        print(self._fillna_fields_dict.items())
        for field_name, dtype in self._fillna_fields_dict.items():
            if dtype in (pymilvus.DataType.VARCHAR, pymilvus.DataType.STRING):
                print("##### 1 1")
                df[field_name] = df[field_name].fillna("")
            elif dtype in (pymilvus.DataType.INT8,
                           pymilvus.DataType.INT16,
                           pymilvus.DataType.INT32,
                           pymilvus.DataType.INT64):
                print("##### 1 2")
                df[field_name] = df[field_name].fillna(0)
            elif dtype in (pymilvus.DataType.FLOAT, pymilvus.DataType.DOUBLE):
                print("##### 1 3")
                df[field_name] = df[field_name].fillna(0.0)
            elif dtype == pymilvus.DataType.BOOL:
                print("##### 1 4")
                df[field_name] = df[field_name].fillna(False)
            else:
                print("##### 1 5")
                logger.info("Skipped checking 'None' in the field: %s, with datatype: %s", field_name, dtype)

        print("##### 2")

        needs_truncate = self._truncate_long_strings
        if needs_truncate and is_cudf_type(df):
            # Cudf specific optimization, we can avoid a costly call to truncate_string_cols_by_bytes if all of the
            # string columns are already below the max length
            needs_truncate = cudf_string_cols_exceed_max_bytes(df, self._fields_max_length)

        # From the schema, this is the list of columns we need, excluding any auto_id columns
        column_names = [field.name for field in self._fields if not field.auto_id]
        print(column_names)

        collection_df = df[column_names]
        if is_cudf_type(collection_df):
            collection_df = collection_df.to_pandas()

        if needs_truncate:
            truncate_string_cols_by_bytes(collection_df, self._fields_max_length, warn_on_truncate=True)

        # Note: dataframe columns has to be in the order of collection schema fields.s
        print(collection_df.to_string(max_rows=3))
        result = self._collection.insert(data=collection_df, **kwargs)
        self._collection.flush()

        return self._insert_result_to_dict(result=result)

    def describe(self, **kwargs: dict[str, typing.Any]) -> dict:
        """
        Provides a description of the collection.

        Parameters
        ----------
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns response content as a dictionary.
        """
        return self._client.describe_collection(collection_name=self._name, **kwargs)

    def query(self, query: str, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Query data in a collection in the Milvus vector database.

        This method performs a search operation in the specified collection/partition in the Milvus vector database.

        Parameters
        ----------
        query : str, optional
            The search query, which can be a filter expression, by default None.
        **kwargs : dict
            Additional keyword arguments for the search operation.

        Returns
        -------
        typing.Any
            The search result, which can vary depending on the query and options.

        Raises
        ------
        RuntimeError
            If an error occurs during the search operation.
            If query argument is `None` and `data` keyword argument doesn't exist.
            If `data` keyword arguement is `None`.
        """

        logger.debug("Searching in collection: %s, query=%s, kwargs=%s", self._name, query, kwargs)

        return self._client.query(collection_name=self._name, filter=query, **kwargs)

    async def similarity_search(self,
                                embeddings: list[list[float]],
                                k: int = 4,
                                **kwargs: dict[str, typing.Any]) -> list[dict]:
        """
        Perform a similarity search within the collection.

        Parameters
        ----------
        embeddings : list[list[float]]
            Embeddings for which to perform the similarity search.
        k : int, optional
            The number of nearest neighbors to return, by default 4.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        list[dict]
            Returns a list of dictionaries representing the results of the similarity search.
        """

        self._collection.load()

        assert self._vector_field is not None, "Cannot perform similarity search on a collection without a vector field"

        # Determine result metadata fields.
        output_fields = [x.name for x in self._fields if x.name != self._vector_field]

        params = {"metric_type": "L2", "params": {"ef": 10}}

        response = self._collection.search(data=embeddings,
                                           anns_field=self._vector_field,
                                           param=params,
                                           limit=k,
                                           output_fields=output_fields,
                                           **kwargs)

        outputs = []

        for res in response:
            outputs.append([{x: hit.entity.get(x) for x in output_fields} for hit in res])

        return outputs

    def update(self, data: list[typing.Any], **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Update data in the collection.

        Parameters
        ----------
        data : list[typing.Any]
            Data to be updated in the collection.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to upsert operation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the updated operation stats.
        """

        if not isinstance(data, list):
            raise RuntimeError("Data is not of type list.")

        result = self._client.upsert(collection_name=self._name, entities=data, **kwargs)

        self._collection.flush()

        return self._update_delete_result_to_dict(result=result)

    def delete_by_keys(self, keys: int | str | list, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Delete vectors by keys from the collection.

        Parameters
        ----------
        keys : int | str | list
            Primary keys to delete vectors.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns result of the given keys that are deleted from the collection.
        """

        result = self._client.delete(collection_name=self._name, pks=keys, **kwargs)

        return result

    def delete(self, expr: str, **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Delete vectors from the collection using expressions.

        Parameters
        ----------
        expr : str
            Delete expression.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the given keys that are deleted from the collection.
        """

        result = self._client.delete_by_expr(collection_name=self._name, expression=expr, **kwargs)

        return self._update_delete_result_to_dict(result=result)

    def retrieve_by_keys(self, keys: int | str | list, **kwargs: dict[str, typing.Any]) -> list[typing.Any]:
        """
        Retrieve the inserted vectors using their primary keys.

        Parameters
        ----------
        keys : int | str | list
            Primary keys to get vectors for. Depending on pk_field type it can be int or str
            or a list of either.
        **kwargs :  dict[str, typing.Any]
            Additional keyword arguments for the retrieval operation.

        Returns
        -------
        list[typing.Any]
            Returns result rows of the given keys from the collection.
        """

        result = None

        try:
            result = self._client.get(collection_name=self._name, ids=keys, **kwargs)
        except pymilvus.exceptions.MilvusException as exec_info:
            raise RuntimeError(f"Unable to perform search: {exec_info}") from exec_info

        return result

    def count(self, **kwargs: dict[str, typing.Any]) -> int:
        """
        Returns number of rows/entities.

        Parameters
        ----------
        **kwargs :  dict[str, typing.Any]
            Additional keyword arguments for the count operation.

        Returns
        -------
        int
            Returns number of entities in the collection.
        """
        return self._collection.num_entities

    def drop(self, **kwargs: dict[str, typing.Any]) -> None:
        """
        Drop a collection, index, or partition in the Milvus vector database.

        This function allows you to drop a collection.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for specifying the type and partition name (if applicable).
        """

        self._collection.drop(**kwargs)

    def _insert_result_to_dict(self, result: "MutationResult") -> dict[str, typing.Any]:
        result_dict = {
            "primary_keys": result.primary_keys,
            "insert_count": result.insert_count,
            "delete_count": result.delete_count,
            "upsert_count": result.upsert_count,
            "timestamp": result.timestamp,
            "succ_count": result.succ_count,
            "err_count": result.err_count,
            "succ_index": result.succ_index,
            "err_index": result.err_index
        }
        return result_dict

    def _update_delete_result_to_dict(self, result: "MutationResult") -> dict[str, typing.Any]:
        result_dict = {
            "insert_count": result.insert_count,
            "delete_count": result.delete_count,
            "upsert_count": result.upsert_count,
            "timestamp": result.timestamp,
            "succ_count": result.succ_count,
            "err_count": result.err_count
        }
        return result_dict


class MilvusVectorDBService(VectorDBService):
    """
    Service class for Milvus Vector Database implementation. This class provides functions for interacting
    with a Milvus vector database.

    Parameters
    ----------
    host : str
        The hostname or IP address of the Milvus server.
    port : str
        The port number for connecting to the Milvus server.
    alias : str, optional
        Alias for the Milvus connection, by default "default".
    truncate_long_strings : bool, optional
        When true, truncate strings values that are longer than the max length of the field
    **kwargs : dict
        Additional keyword arguments specific to the Milvus connection configuration.
    """

    _collection_locks = {}
    _cleanup_interval = 600  # 10mins
    _last_cleanup_time = time.time()

    def __init__(self,
                 uri: str,
                 user: str = "",
                 password: str = "",
                 db_name: str = "",
                 token: str = "",
                 truncate_long_strings: bool = False,
                 **kwargs: dict[str, typing.Any]):

        self._truncate_long_strings = truncate_long_strings
        self._client = MilvusClient(uri=uri, user=user, password=password, db_name=db_name, token=token, **kwargs)

    def load_resource(self, name: str, **kwargs: dict[str, typing.Any]) -> MilvusVectorDBResourceService:
        return MilvusVectorDBResourceService(name=name,
                                             client=self._client,
                                             truncate_long_strings=self._truncate_long_strings,
                                             **kwargs)

    def has_store_object(self, name: str) -> bool:
        """
        Check if a collection exists in the Milvus vector database.

        Parameters
        ----------
        name : str
            Name of the collection to check.

        Returns
        -------
        bool
            True if the collection exists, False otherwise.
        """
        return self._client.has_collection(collection_name=name)

    def list_store_objects(self, **kwargs: dict[str, typing.Any]) -> list[str]:
        """
        List the names of all collections in the Milvus vector database.

        Returns
        -------
        list[str]
            A list of collection names.
        """
        return self._client.list_collections(**kwargs)

    def _create_schema_field(self, field_conf: dict) -> "pymilvus.FieldSchema":

        field_schema = pymilvus.FieldSchema.construct_from_dict(field_conf)

        return field_schema

    @with_collection_lock
    def create(self, name: str, overwrite: bool = False, **kwargs: dict[str, typing.Any]):
        """
        Create a collection in the Milvus vector database with the specified name and configuration. This method
        creates a new collection in the Milvus vector database with the provided name and configuration options.
        If the collection already exists, it can be overwritten if the `overwrite` parameter is set to True.

        Parameters
        ----------
        name : str
            Name of the collection to be created.
        overwrite : bool, optional
            If True, the collection will be overwritten if it already exists, by default False.
        **kwargs : dict
            Additional keyword arguments containing collection configuration.

        Raises
        ------
        ValueError
            If the provided schema fields configuration is empty.
        """
        logger.debug("Creating collection: %s, overwrite=%s, kwargs=%s", name, overwrite, kwargs)

        # Preserve original configuration.
        collection_conf = copy.deepcopy(kwargs)

        auto_id = collection_conf.get("auto_id", False)
        index_conf = collection_conf.get("index_conf", None)
        partition_conf = collection_conf.get("partition_conf", None)

        schema_conf = collection_conf.get("schema_conf")
        schema_fields_conf = schema_conf.pop("schema_fields")

        if not self.has_store_object(name) or overwrite:
            if overwrite and self.has_store_object(name):
                self.drop(name)

            if len(schema_fields_conf) == 0:
                raise ValueError("Cannot create collection as provided empty schema_fields configuration")

            schema_fields = [FieldSchemaEncoder.from_dict(field_conf) for field_conf in schema_fields_conf]

            schema = pymilvus.CollectionSchema(fields=schema_fields, **schema_conf)

            self._client.create_collection_with_schema(collection_name=name,
                                                       schema=schema,
                                                       index_params=index_conf,
                                                       auto_id=auto_id,
                                                       shards_num=collection_conf.get("shards", 2),
                                                       consistency_level=collection_conf.get(
                                                           "consistency_level", "Strong"))

            if partition_conf:
                timeout = partition_conf.get("timeout", 1.0)
                # Iterate over each partition configuration
                for part in partition_conf["partitions"]:
                    self._client.create_partition(collection_name=name, partition_name=part["name"], timeout=timeout)

    def _build_schema_conf(self, df: DataFrameType) -> list[dict]:
        fields = []

        # Always add a primary key
        fields.append({"name": "pk", "dtype": pymilvus.DataType.INT64, "is_primary": True, "auto_id": True})

        if is_cudf_type(df):
            df = df.to_pandas()

        # Loop over all of the columns of the first row and build the schema
        for col_name, col_val in df.iloc[0].items():

            field_dict = {
                "name": col_name,
                "dtype": pymilvus.orm.types.infer_dtype_bydata(col_val),
                # "is_primary": col_name == kwargs.get("primary_key", None),
                # "auto_id": col_name == kwargs.get("primary_key", None)
            }

            if (field_dict["dtype"] == pymilvus.DataType.VARCHAR):
                field_dict["max_length"] = MAX_STRING_LENGTH_BYTES

            if (field_dict["dtype"] == pymilvus.DataType.FLOAT_VECTOR
                    or field_dict["dtype"] == pymilvus.DataType.BINARY_VECTOR):
                field_dict["params"] = {"dim": len(col_val)}

            if (field_dict["dtype"] == pymilvus.DataType.UNKNOWN):
                logger.warning("Could not infer data type for column '%s', with value: %s. Skipping column in schema.",
                               col_name,
                               col_val)
                continue

            fields.append(field_dict)

        return fields

    def create_from_dataframe(self,
                              name: str,
                              df: DataFrameType,
                              overwrite: bool = False,
                              **kwargs: dict[str, typing.Any]) -> None:
        """
        Create collections in the vector database.

        Parameters
        ----------
        name : str
            Name of the collection.
        df : DataFrameType
            The dataframe to create the collection from.
        overwrite : bool, optional
            Whether to overwrite the collection if it already exists. Default is False.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.
        """

        fields = self._build_schema_conf(df=df)

        create_kwargs = {
            "schema_conf": {
                "description": "Auto generated schema from DataFrame in Morpheus",
                "schema_fields": fields,
            }
        }

        if (kwargs.get("index_field", None) is not None):
            # Check to make sure the column name exists in the fields
            create_kwargs["index_conf"] = {
                "field_name": kwargs.get("index_field"),  # Default index type
                "metric_type": "L2",
                "index_type": "HNSW",
                "params": {
                    "M": 8,
                    "efConstruction": 64,
                },
            }

        self.create(name=name, overwrite=overwrite, **create_kwargs)

    @with_collection_lock
    def insert(self, name: str, data: list[list] | list[dict], **kwargs: dict[str,
                                                                              typing.Any]) -> dict[str, typing.Any]:
        """
        Insert a collection specific data in the Milvus vector database.

        Parameters
        ----------
        name : str
            Name of the collection to be inserted.
        data : list[list] | list[dict]
            Data to be inserted in the collection.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments containing collection configuration.

        Returns
        -------
        dict
            Returns response content as a dictionary.

        Raises
        ------
        RuntimeError
            If the collection not exists exists.
        """

        resource = self.load_resource(name)
        return resource.insert(data, **kwargs)

    @with_collection_lock
    def insert_dataframe(self, name: str, df: DataFrameType, **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Converts dataframe to rows and insert to a collection in the Milvus vector database.

        Parameters
        ----------
        name : str
            Name of the collection to be inserted.
        df : DataFrameType
            Dataframe to be inserted in the collection.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments containing collection configuration.

        Returns
        -------
        dict
            Returns response content as a dictionary.

        Raises
        ------
        RuntimeError
            If the collection not exists exists.
        """
        resource = self.load_resource(name)
        print(name)
        print(resource)

        return resource.insert_dataframe(df=df, **kwargs)

    @with_collection_lock
    def query(self, name: str, query: str = None, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Query data in a collection in the Milvus vector database.

        This method performs a search operation in the specified collection/partition in the Milvus vector database.

        Parameters
        ----------
        name : str
            Name of the collection to search within.
        query : str
            The search query, which can be a filter expression.
        **kwargs : dict
            Additional keyword arguments for the search operation.

        Returns
        -------
        typing.Any
            The search result, which can vary depending on the query and options.
        """

        resource = self.load_resource(name)

        return resource.query(query, **kwargs)

    async def similarity_search(self, name: str, **kwargs: dict[str, typing.Any]) -> list[dict]:
        """
        Perform a similarity search within the collection.

        Parameters
        ----------
        name : str
            Name of the collection.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        list[dict]
            Returns a list of dictionaries representing the results of the similarity search.
        """

        resource = self.load_resource(name)

        return resource.similarity_search(**kwargs)

    @with_collection_lock
    def update(self, name: str, data: list[typing.Any], **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Update data in the vector database.

        Parameters
        ----------
        name : str
            Name of the collection.
        data : list[typing.Any]
            Data to be updated in the collection.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to upsert operation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the updated operation stats.
        """

        if not isinstance(data, list):
            raise RuntimeError("Data is not of type list.")

        resource = self.load_resource(name)

        return resource.update(data=data, **kwargs)

    @with_collection_lock
    def delete_by_keys(self, name: str, keys: int | str | list, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Delete vectors by keys from the collection.

        Parameters
        ----------
        name : str
            Name of the collection.
        keys : int | str | list
            Primary keys to delete vectors.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns result of the given keys that are delete from the collection.
        """

        resource = self.load_resource(name)

        return resource.delete_by_keys(keys=keys, **kwargs)

    @with_collection_lock
    def delete(self, name: str, expr: str, **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Delete vectors from the collection using expressions.

        Parameters
        ----------
        name : str
            Name of the collection.
        expr : str
            Delete expression.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the given keys that are delete from the collection.
        """

        resource = self.load_resource(name)
        result = resource.delete(expr=expr, **kwargs)

        return result

    @with_collection_lock
    def retrieve_by_keys(self, name: str, keys: int | str | list, **kwargs: dict[str, typing.Any]) -> list[typing.Any]:
        """
        Retrieve the inserted vectors using their primary keys from the Collection.

        Parameters
        ----------
        name : str
            Name of the collection.
        keys : int | str | list
            Primary keys to get vectors for. Depending on pk_field type it can be int or str
            or a list of either.
        **kwargs :  dict[str, typing.Any]
            Additional keyword arguments for the retrieval operation.

        Returns
        -------
        list[typing.Any]
            Returns result rows of the given keys from the collection.
        """

        resource = self.load_resource(name)

        result = resource.retrieve_by_keys(keys=keys, **kwargs)

        return result

    def count(self, name: str, **kwargs: dict[str, typing.Any]) -> int:
        """
        Returns number of rows/entities in the given collection.

        Parameters
        ----------
        name : str
            Name of the collection.
        **kwargs :  dict[str, typing.Any]
            Additional keyword arguments for the count operation.

        Returns
        -------
        int
            Returns number of entities in the collection.
        """
        resource = self.load_resource(name)

        return resource.count(**kwargs)

    def drop(self, name: str, **kwargs: dict[str, typing.Any]) -> None:
        """
        Drop a collection, index, or partition in the Milvus vector database.

        This method allows you to drop a collection, an index within a collection,
        or a specific partition within a collection in the Milvus vector database.

        Parameters
        ----------
        name : str
            Name of the collection, index, or partition to be dropped.
        **kwargs : dict
            Additional keyword arguments for specifying the type and partition name (if applicable).

        Notes on Expected Keyword Arguments:
        ------------------------------------
        - 'collection' (str, optional):
        Specifies the type of collection to drop. Possible values: 'collection' (default), 'index', 'partition'.

        - 'partition_name' (str, optional):
        Required when dropping a specific partition within a collection. Specifies the partition name to be dropped.

        - 'field_name' (str, optional):
        Required when dropping an index within a collection. Specifies the field name for which the index is created.

        - 'index_name' (str, optional):
        Required when dropping an index within a collection. Specifies the name of the index to be dropped.

        Raises
        ------
        ValueError
            If mandatory arguments are missing or if the provided 'collection' value is invalid.
        """

        logger.debug("Dropping collection: %s, kwargs=%s", name, kwargs)

        if self.has_store_object(name):
            resource = kwargs.get("resource", "collection")
            if resource == "collection":
                self._client.drop_collection(collection_name=name)
            elif resource == "partition":
                if "partition_name" not in kwargs:
                    raise ValueError("Mandatory argument 'partition_name' is required when resource='partition'")
                partition_name = kwargs["partition_name"]
                if self._client.has_partition(collection_name=name, partition_name=partition_name):
                    # Collection need to be released before dropping the partition.
                    self._client.release_collection(collection_name=name)
                    self._client.drop_partition(collection_name=name, partition_name=partition_name)
            elif resource == "index":
                if "field_name" in kwargs and "index_name" in kwargs:
                    self._client.drop_index(collection_name=name,
                                            field_name=kwargs["field_name"],
                                            index_name=kwargs["index_name"])
                else:
                    raise ValueError(
                        "Mandatory arguments 'field_name' and 'index_name' are required when resource='index'")

    def describe(self, name: str, **kwargs: dict[str, typing.Any]) -> dict:
        """
        Describe the collection in the vector database.

        Parameters
        ----------
        name : str
            Name of the collection.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments specific to the Milvus vector database.

        Returns
        -------
        dict
            Returns collection information.
        """

        resource = self.load_resource(name)

        return resource.describe(**kwargs)

    def release_resource(self, name: str) -> None:
        """
        Release a loaded collection from the memory.

        Parameters
        ----------
        name : str
            Name of the collection to release.
        """

        self._client.release_collection(collection_name=name)

    def close(self) -> None:
        """
        Close the connection to the Milvus vector database.

        This method disconnects from the Milvus vector database by removing the connection.

        """
        self._client.close()

    @classmethod
    def get_collection_lock(cls, name: str) -> threading.Lock:
        """
        Get a lock for a given collection name.

        Parameters
        ----------
        name : str
            Name of the collection for which to acquire the lock.

        Returns
        -------
        threading.Lock
            A thread lock specific to the given collection name.
        """

        current_time = time.time()

        if name not in cls._collection_locks:
            cls._collection_locks[name] = {"lock": threading.Lock(), "last_used": current_time}
        else:
            cls._collection_locks[name]["last_used"] = current_time

        if (current_time - cls._last_cleanup_time) >= cls._cleanup_interval:
            for lock_name, lock_info in cls._collection_locks.copy().items():
                last_used = lock_info["last_used"]
                if current_time - last_used >= cls._cleanup_interval:
                    logger.debug("Cleaning up lock for collection: %s", lock_name)
                    del cls._collection_locks[lock_name]
            cls._last_cleanup_time = current_time

        return cls._collection_locks[name]["lock"]
