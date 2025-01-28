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

import copy
import enum
import json
import logging
import threading
import time
import typing
from collections import OrderedDict
from functools import wraps

from gpudb import GPUdb, GPUdbTable, GPUdbRecordColumn, GPUdbRecordType, GPUdbException, GPUdbSqlIterator
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Token
from sqlparse.tokens import Keyword

from morpheus.io.utils import cudf_string_cols_exceed_max_bytes
from morpheus.io.utils import truncate_string_cols_by_bytes
from morpheus.utils.type_aliases import DataFrameType
from morpheus.utils.type_utils import is_cudf_type
from morpheus_llm.error import IMPORT_ERROR_MESSAGE
from morpheus_llm.service.vdb.vector_db_service import VectorDBResourceService
from morpheus_llm.service.vdb.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None

try:
    import gpudb

except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc


class DistanceStrategy(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    EUCLIDEAN = "l2"
    COSINE = "cosine"
    MAX_INNER_PRODUCT = "inner"

class Dimension(int, enum.Enum):
    """Some default dimensions for known embeddings."""

    OPENAI = 1536


DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.EUCLIDEAN

class KineticaVectorDBResourceService(VectorDBResourceService):
    """
    Represents a service for managing resources in a Kinetica Vector Database.

    Parameters
    ----------
    name : str
        Name of the Kinetica table. Must be an existing table
    schema : str
        Name of the Kinetica schema. Must be an existing schema
    client : GPUdb instance
        An instance of the GPUdb class for interaction with the Kinetica Vector Database.
    """

    def __init__(self, name: str, client: "GPUdb") -> None:
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE.format(package='gpudb')) from IMPORT_EXCEPTION

        super().__init__()

        self._name = name
        self._client = client

        self._collection =  GPUdbTable(name=self._name, db=client)
        self._record_type = self._collection.get_table_type()
        self._fields: list[GPUdbRecordColumn] = self._record_type.columns
        self._description = self.describe()

        self._vector_field = None
        self._fillna_fields_dict = {}

        # Mapping of field name to max length for string fields
        self._fields_max_length: dict[str, int] = {}

        for field in self._fields:
            if field.is_vector():
                self._vector_field = field.name
            else:
                if not field.column_properties[1] == "primary_key":
                    self._fillna_fields_dict[field.name] = field.column_type


    def insert(self, data: list[list] | list[dict], **kwargs: dict[str, typing.Any]) -> dict:
        """
        Insert data into the vector database.

        Parameters
        ----------
        data : list[list] | list[dict]
            Data to be inserted into the Kinetica table.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns response content as a dictionary.
        """
        options = kwargs.get( "options", None )
        if options is not None: # if given, remove from kwargs
            kwargs.pop( "options" )
        else: # no option given; use an empty dict
            options = {}

        result = self._collection.insert_records(data, options=options)

        return self._insert_result_to_dict(result=result.count)

    def insert_dataframe(self, df: DataFrameType, **kwargs: dict[str, typing.Any]) -> dict:
        """
        Insert a dataframe entires into the vector database.

        Parameters
        ----------
        df : DataFrameType
            Dataframe to be inserted into the Kinetica table.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns response content as a dictionary.
        """
        # From the schema, this is the list of columns we need
        column_names = [field.name for field in self._fields]

        collection_df = df[column_names]
        if is_cudf_type(collection_df):
            collection_df = collection_df.to_pandas()

        # Note: dataframe columns has to be in the order of Kinetica table schema fields.s
        result = self._collection.insert_df(collection_df)

        return self._insert_result_to_dict(result=result)

    def describe(self, **kwargs: dict[str, typing.Any]) -> dict:
        """
        Provides a description of the Kinetica table.

        Parameters
        ----------
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns response content as a dictionary.
        """
        table1 = GPUdbTable(db=self._client, name=self._name)
        record_type: GPUdbRecordType = table1.get_table_type()

        col: GPUdbRecordColumn = None
        description: dict[str, object] = {}

        for col in record_type.columns:
            description[col.name] = (
                col.column_type,
                col.is_nullable,
                col.is_vector(),
                col.column_properties,
                "primary_key" if "primary_key" in col.column_properties else ""
            )

        return description

    def _get_pk_field_name(self):
        for field_name, properties in self._description.items():
            if properties[4] == "primary_key":
                return field_name

        return ""

    def query(self, query: str, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Query data in a table in the Kinetica database.

        This method performs a search operation in the specified table in the Kinetica database.

        Parameters
        ----------
        query : str, optional
            The search query, which is an SQL query.
        **kwargs : dict
            Additional keyword arguments for the search operation.

        Returns
        -------
        typing.Any
            The search result (a GPUdbSqlIterator object)

        Raises
        ------
        GPUdbException
            If an error occurs during the search operation.
        """

        if query is None:
            raise GPUdbException("'query' - a valid SQL query statement must be given ...")

        logger.debug("Searching in Kinetica table: %s, query=%s, kwargs=%s", self._name, query, kwargs)
        batch_size = kwargs.get("batch_size", 5000)
        sql_params = kwargs.get("sql_params", [])
        sql_opts = kwargs.get("sql_opts", {})
        return self._client.query(query, batch_size, sql_params, sql_opts)

    def _extract_projection_fields(self, parsed_tokens):
        """
        Recursively extracts projection fields from the SQL tokens.

        Parameters:
            parsed_tokens (list): List of tokens from a parsed SQL query.

        Returns:
            list: A list of field names in the projection list.
        """
        fields = []
        for token in parsed_tokens:
            if isinstance(token, IdentifierList):
                # Multiple fields in the projection list
                for identifier in token.get_identifiers():
                    fields.append(identifier.get_real_name())
            elif isinstance(token, Identifier):
                # Single field
                fields.append(token.get_real_name())
            elif token.ttype is Keyword and token.value.upper() in ["SELECT"]:
                # Skip the SELECT keyword
                continue
            elif token.is_group:
                # Handle nested subqueries
                fields.extend(self._extract_projection_fields(token.tokens))
        return fields

    def _is_field_in_projection(self, sql_statement, field_name):
        """
        Check whether a given field name is present in the projection list of an SQL statement.

        Parameters:
            sql_statement (str): The SQL query to parse.
            field_name (str): The field name to search for.

        Returns:
            bool: True if the field name is present in the projection list, False otherwise.
        """
        parsed = sqlparse.parse(sql_statement)
        for stmt in parsed:
            if stmt.get_type() == "SELECT":
                # Extract projection fields
                projection_fields = self._extract_projection_fields(stmt.tokens)
                # Check if the field is in the projection list
                return field_name in projection_fields
        return False

    def __query_collection(
        self,
        embedding: list[float],
        output_fields: list[str],
        k: int = 4,
        filter: dict[str, str] = None,
    ) -> dict:
        """Query the Kinetica table."""
        # if filter is not None:
        #     filter_clauses = []
        #     for key, value in filter.items():
        #         IN = "in"
        #         if isinstance(value, dict) and IN in map(str.lower, value):
        #             value_case_insensitive = {
        #                 k.lower(): v for k, v in value.items()
        #             }
        #             filter_by_metadata = self.EmbeddingStore.cmetadata[
        #                 key
        #             ].astext.in_(value_case_insensitive[IN])
        #             filter_clauses.append(filter_by_metadata)
        #         else:
        #             filter_by_metadata = self.EmbeddingStore.cmetadata[
        #                 key
        #             ].astext == str(value)
        #             filter_clauses.append(filter_by_metadata)

        json_filter = json.dumps(filter) if filter is not None else None
        where_clause = (
            f" where '{json_filter}' = JSON(metadata) "
            if json_filter is not None
            else ""
        )

        embedding_str = "[" + ",".join([str(x) for x in embedding]) + "]"

        dist_strategy = DEFAULT_DISTANCE_STRATEGY

        query_string = f"""
                SELECT {', '.join(output_fields)}, {dist_strategy}(embedding, '{embedding_str}') 
                as distance, {self._vector_field}
                FROM {self._collection_name}
                {where_clause}
                ORDER BY distance asc NULLS LAST
                LIMIT {k}
        """

        self.logger.debug(query_string)
        resp = self._client.execute_sql_and_decode(query_string)
        self.logger.debug(resp)
        return resp

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        output_fields: list[str],
        k: int = 4,
        filter: dict = None,
        **kwargs: typing.Any,
    ) -> list[dict]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List of records most similar to the query vector.
        """
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, output_fields=output_fields, k=k, filter=filter
        )
        return docs

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        output_fields: list[str],
        k: int = 4,
        filter: dict = None,
    ) -> list[dict]:

        resp: dict = self.__query_collection(embedding, k, filter)
        if resp and resp["status_info"]["status"] == "OK" and "records" in resp:
            records: OrderedDict = resp["records"]
            return [records]

        self.logger.error(resp["status_info"]["message"])
        return []


    async def similarity_search(self,
                                embeddings: list[list[float]],
                                k: int = 4,
                                **kwargs: dict[str, typing.Any]) -> list[list[dict]]:
        """
        Perform a similarity search within the Kinetica table.

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
        list[list[dict]]
            Returns a list of 'list of dictionaries' representing the results of the similarity search.
        """

        assert self._vector_field is not None, "Cannot perform similarity search on a table without a vector field"

        # Determine result metadata fields.
        output_fields = [x.name for x in self._fields if x.name != self._vector_field]
        search_filter = kwargs.get("filter", "")

        results: list[list[dict]] = [self.similarity_search_by_vector(
            embedding=embedding,
            output_fields=output_fields,
            k=k,
            filter=search_filter,
        ) for embedding in embeddings]

        return results

    def update(self, data: list[typing.Any], **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Update data in the Kinetica table.

        Parameters
        ----------
        data : list[typing.Any]
            Data to be updated in the Kinetica table. This is npt used by Kinetica. The required parameters
            are all passed in a keyword arguments.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to Kinetica. The full explanation of each of these
            keyword arguments is available in the documentation of the API `/update/records`.

            Allowed keyword arguments:
                options: dict[str, str] - options

                expressions: [list] - expression used to filter records for update

                new_value_maps: [list[dict]] | [dict] -

                records_to_insert: list[] -

                records_to_insert_str: list[] -



        Returns
        -------
        dict[str, typing.Any]
            Returns result of the updated operation stats.
        """

        options = kwargs.get( "options", None )
        if options is not None: # if given, remove from kwargs
            kwargs.pop( "options" )
        else: # no option given; use an empty dict
            options = {}

        expressions = kwargs.get( "expressions", [] )
        if expressions is not None: # if given, remove from kwargs
            if not isinstance(expressions, list):
                raise GPUdbException("'expressions' must be of type 'list' ...")
            kwargs.pop( "expressions" )
        else: # no option given; use an empty dict
            raise GPUdbException("Update 'expressions' must be given ...")

        new_values_maps = kwargs.get( "new_values_maps", None )
        if new_values_maps is not None: # if given, remove from kwargs
            if not isinstance(new_values_maps, (list, dict)):
                raise GPUdbException("'new_value_maps' should either be a 'list of dicts' or a dict ...")
            kwargs.pop( "new_values_maps" )
        else: # no option given; use an empty dict
            raise GPUdbException("'new_values_maps' must be given ...")

        if len(expressions) != len(new_values_maps):
            raise GPUdbException("'expression' and 'new_value_maps' must have the same number of elements")

        records_to_insert = kwargs.get("records_to_insert", [])
        records_to_insert_str = kwargs.get("records_to_insert_str", [])

        result = self._collection.update_records(expressions, new_values_maps, records_to_insert, records_to_insert_str, options=options)

        return self._update_delete_result_to_dict(result=result)

    def delete(self, expr: str, **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Delete vectors from the Kinetica table using expressions.

        Parameters
        ----------
        expr : str
            Delete expression.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the given keys that are deleted from the Kinetica table.
        """

        result = self._collection.delete_records(expressions=[expr], **kwargs)

        return self._update_delete_result_to_dict(result=result)

    def delete_by_keys(self, keys: int | str | list, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Delete vectors by keys from the resource.

        Parameters
        ----------
        keys : int | str | list
            Primary keys to delete vectors.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns vectors of the given keys that are delete from the resource.
        """
        pass

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

            Only valid keyword arguments are:

                expression: [str] - The Kinetica expression to pass on to `get/records/by/key`

                options: dict - The `options` dict accepted by `get/records/by/key`

        Returns
        -------
        list[typing.Any]
            Returns result rows of the given keys from the Kinetica table.
        """

        result = None
        expression = kwargs.get("expression", "")
        options = kwargs.get("options", {})
        if keys is None or keys <= 0 or (isinstance(keys, (str, list)) and len(keys) <= 0):
            raise GPUdbException("'keys' must be specified as either an 'int' or 'str' or 'list' ...")

        if expression == "":
            # expression not specified; keys must be values of PK in the Kinetica table.
            pk_field_name = self._get_pk_field_name()
            if pk_field_name == "":
                raise GPUdbException("No 'expression' given and no 'PK field' found cannot retrieve records ...")

            if isinstance(keys, str):
                expression = [f"{pk_field_name} = '{keys}'"]
            elif isinstance(keys, int):
                expression = [f"{pk_field_name} = {keys}"]
            elif isinstance(keys, list):
                # keys is a list
                expression = [f"{pk_field_name} in ({','.join(keys)})"]
            else:
                raise GPUdbException("'keys' must be of type (int or str or list) ...")
        try:
            result = self._collection.get_records_by_key(keys, expression, options)
        except GPUdbException as exec_info:
            raise RuntimeError(f"Unable to perform search: {exec_info}") from exec_info

        return result["records"] if result is not None and 'records' in result else []

    def count(self, **kwargs: dict[str, typing.Any]) -> int:
        """
        Returns number of rows/entities.

        Parameters
        ----------
        **kwargs :  dict[str, typing.Any]
            Not used.

        Returns
        -------
        int
            Returns number of records in the Kinetica table.
        """
        return self._collection.count

    def drop(self, **kwargs: dict[str, typing.Any]) -> None:
        """
        Drop a Kinetica table.

        This function allows you to delete/drop a Kinetica table.

        Parameters
        ----------
        **kwargs : dict
            Not used.
        """

        self._client.clear_table(self._collection_name)

    def _insert_result_to_dict(self, result: int) -> dict[str, typing.Any]:
        result_dict = {
            "count": result,
        }
        return result_dict

    def _update_delete_result_to_dict(self, result) -> dict[str, typing.Any]:
        result_dict = {
            "count_deleted": result["count_deleted"],
            "counts_updated": result["counts_deleted"],
            "info": result["info"],
        }
        return result_dict


class KineticaVectorDBService(VectorDBService):
    """
    Service class for Kinetica Database implementation. This class provides functions for interacting
    with a Kinetica database.

    Parameters
    ----------
    host : str
        The hostname or IP address of the Kinetica server.
    port : str
        The port number for connecting to the Kinetica server.
    alias : str, optional
        Alias for the Kinetica connection, by default "default".
    """

    def __init__(self,
                 uri: str,
                 user: str = "",
                 password: str = "",
                 kinetica_schema = "",
                 ):
        options = GPUdb.Options()
        options.skip_ssl_cert_verification = True
        options.username = user
        options.password = password

        self._collection_name = None
        self.schema = kinetica_schema if kinetica_schema is not None and len(kinetica_schema) > 0 else None
        self._client = GPUdb(host=uri, options=options)

    def load_resource(self, name: str, **kwargs: dict[str, typing.Any]) -> KineticaVectorDBResourceService:
        """

        @param name:
        @param kwargs:
        @return:
        """
        self._collection_name = f"{self.schema}.{name}" if self.schema is not None and len(self.schema) > 0 else f"ki_home.{name}"
        return KineticaVectorDBResourceService(name=self._collection_name,
                                               client=self._client)

    @property
    def collection_name(self):
        return self._collection_name if self._collection_name is not None else None

    def has_store_object(self, name: str) -> bool:
        """
        Check if a table exists in the Kinetica database.

        Parameters
        ----------
        name : str
            Name of the table to check.

        Returns
        -------
        bool
            True if the table exists, False otherwise.
        """
        return self._client.has_table(name)["table_exists"]

    def create(self, name: str, overwrite: bool = False, **kwargs: dict[str, typing.Any]):
        """
        Create a table in the Kinetica database with the specified name and configuration.
        If the table already exists, it can be overwritten if the `overwrite` parameter is set to True.

        Parameters
        ----------
        name : str
            Name of the table to be created. The name must be in the form 'schema_name.table_name'.
        overwrite : bool, optional
            If True, the Kinetica table will be overwritten if it already exists, by default False.
        **kwargs : dict
            Additional keyword arguments containing Kinetica `/create/table` options.

        Raises
        ------
        GPUdbException
            If the provided type schema configuration is empty.
        """
        logger.debug("Creating Kinetica table: %s, overwrite=%s, kwargs=%s", name, overwrite, kwargs)

        table_type: list[list[str]] = kwargs.get("type", [])
        if not self.has_store_object(name) and (table_type is None or len(table_type) == 0):
            raise GPUdbException("Table must either be existing or a type must be given to create the table ...")

        options = kwargs.get("options", {})
        if len(options) == 0:
            options['no_error_if_exists'] = 'true'

        if not self.has_store_object(name) or overwrite:
            if overwrite and self.has_store_object(name):
                self.drop(name)

            GPUdbTable(table_type, name, options=options, db=self._client)


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
            Name of the Kinetica table.
        df : DataFrameType
            The dataframe to create the Kinetica table from.
        overwrite : bool, optional
            Whether to overwrite the Kinetica table if it already exists. Default is False.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.
        """

        GPUdbTable.from_df(df, self._client, name, clear_table=overwrite)

    def insert(self, name: str, data: list[list] | list[dict], **kwargs: dict[str,
                                                                              typing.Any]) -> dict[str, typing.Any]:
        """
        Insert a collection specific data in the Kinetica vector database.

        Parameters
        ----------
        name : str
            Name of the Kinetica table to be inserted.
        data : list[list] | list[dict]
            Data to be inserted in the Kinetica table.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments containing Kinetica table configuration.

        Returns
        -------
        dict
            Returns response content as a dictionary.

        Raises
        ------
        RuntimeError
            If the table not exists.
        """

        resource = self.load_resource(name)
        return resource.insert(data, **kwargs)

    def insert_dataframe(self, name: str, df: DataFrameType, **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Converts dataframe to rows and insert to a Kinetica table in the Kinetica vector database.

        Parameters
        ----------
        name : str
            Name of the Kinetica table to be inserted.
        df : DataFrameType
            Dataframe to be inserted in the Kinetica table.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments containing Kinetica table configuration.

        Returns
        -------
        dict
            Returns response content as a dictionary.

        Raises
        ------
        RuntimeError
            If the Kinetica table not exists.
        """
        resource = self.load_resource(name)

        return resource.insert_dataframe(df=df, **kwargs)

    def query(self, name: str, query: str = None, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Query data in a Kinetica table in the Kinetica vector database.

        This method performs a search operation in the specified Kinetica table/partition in the Kinetica vector database.

        Parameters
        ----------
        name : str
            Name of the Kinetica table to search within.
        query : str
            The search query, which is an SQL query.
        **kwargs : dict
            Additional keyword arguments for the search operation.

        Returns
        -------
        typing.Any
            The search result, which can vary depending on the query and options.
        """

        resource = self.load_resource(name)

        return resource.query(query, **kwargs)

    async def similarity_search(self, name: str, **kwargs: dict[str, typing.Any]) -> list[list[dict]]:
        """
        Perform a similarity search within the Kinetica table.

        Parameters
        ----------
        name : str
            Name of the Kinetica table.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        list[dict]
            Returns a list of dictionaries representing the results of the similarity search.
        """

        resource = self.load_resource(name)

        return resource.similarity_search(**kwargs)

    def update(self, name: str, data: list[typing.Any], **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Update data in the vector database.

        Parameters
        ----------
        name : str
            Name of the Kinetica table.
        data : list[typing.Any]
            Data to be updated in the Kinetica table.
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

    def delete(self, name: str, expr: str, **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Delete vectors from the Kinetica table using expressions.

        Parameters
        ----------
        name : str
            Name of the Kinetica table.
        expr : str
            Delete expression.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the given keys that are delete from the Kinetica table.
        """

        resource = self.load_resource(name)
        result = resource.delete(expr=expr, **kwargs)

        return result

    def retrieve_by_keys(self, name: str, keys: int | str | list, **kwargs: dict[str, typing.Any]) -> list[typing.Any]:
        """
        Retrieve the inserted vectors using their primary keys from the Kinetica table.

        Parameters
        ----------
        name : str
            Name of the Kinetica table.
        keys : int | str | list
            Primary keys to get vectors for. Depending on pk_field type it can be int or str
            or a list of either.
        **kwargs :  dict[str, typing.Any]
            Additional keyword arguments for the retrieval operation.

        Returns
        -------
        list[typing.Any]
            Returns result rows of the given keys from the Kinetica table.
        """

        resource = self.load_resource(name)

        result = resource.retrieve_by_keys(keys=keys, **kwargs)

        return result

    def count(self, name: str, **kwargs: dict[str, typing.Any]) -> int:
        """
        Returns number of rows/entities in the given Kinetica table.

        Parameters
        ----------
        name : str
            Name of the Kinetica table.
        **kwargs :  dict[str, typing.Any]
            Additional keyword arguments for the count operation.

        Returns
        -------
        int
            Returns number of entities in the Kinetica table.
        """
        resource = self.load_resource(name)

        return resource.count(**kwargs)

    def drop(self, name: str, **kwargs: dict[str, typing.Any]) -> None:
        """
        Drop a table in the Kinetica database.

        This method allows you to drop a table in the Kinetica database.

        Parameters
        ----------
        name : str
            Name of the table
        **kwargs : dict
            Additional keyword arguments for specifying the type and partition name (if applicable).

        Notes on Expected Keyword Arguments:
        ------------------------------------
        - 'schema' (str, optional):
        Specifies the schema of the table to drop. Default 'ki_home'

        Raises
        ------
        ValueError
            If mandatory arguments are missing or if the provided 'Kinetica table' value is invalid.
        """

        logger.debug("Dropping Kinetica table: %s, kwargs=%s", name, kwargs)

        if self.has_store_object(name):
            # schema = kwargs.get("schema", "ki_home")
            try:
                self._client.clear_table(name)
            except GPUdbException as e:
                raise ValueError(e.message)

    def describe(self, name: str, **kwargs: dict[str, typing.Any]) -> dict:
        """
        Describe the Kinetica table in the vector database.

        Parameters
        ----------
        name : str
            Name of the Kinetica table.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments specific to the Kinetica vector database.

        Returns
        -------
        dict
            Returns Kinetica table information.
        """

        resource = self.load_resource(name)

        return resource.describe(**kwargs)

    def release_resource(self, name: str) -> None:
        """
        Release a loaded resource from the memory.

        Parameters
        ----------
        name : str
            Name of the resource to release.
        """
        pass

    def close(self) -> None:
        """
        Close connection to the vector database.
        """

        pass

    def list_store_objects(self, **kwargs: dict[str, typing.Any]) -> list[str]:
        """
        List existing resources in the vector database.

        Parameters
        ----------
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        list[str]
            Returns available resouce names in the vector database.
        """

        pass

    def delete_by_keys(self, keys: int | str | list, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Delete vectors by keys from the resource.

        Parameters
        ----------
        keys : int | str | list
            Primary keys to delete vectors.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns vectors of the given keys that are delete from the resource.
        """
        pass
