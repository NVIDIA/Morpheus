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

import asyncio
import logging
import time
import typing

from morpheus.utils.type_aliases import DataFrameType
from morpheus_llm.error import IMPORT_ERROR_MESSAGE
from morpheus_llm.service.vdb.vector_db_service import VectorDBResourceService
from morpheus_llm.service.vdb.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)

IMPORT_EXCEPTION = None

try:
    from langchain.embeddings.base import Embeddings
    from langchain.vectorstores.faiss import FAISS
except ImportError as import_exc:
    IMPORT_EXCEPTION = import_exc


class FaissVectorDBResourceService(VectorDBResourceService):
    """
    Represents a service for managing resources in a FAISS Vector Database.

    Parameters
    ----------
    parent : FaissVectorDBService
        The parent service for this resource.
    name : str
        The name of the resource.
    """

    def __init__(self, parent: "FaissVectorDBService", *, name: str) -> None:
        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE.format(package='langchain and faiss-gpu')) from IMPORT_EXCEPTION

        super().__init__()

        self._parent = parent
        self._folder_path = self._parent._local_dir
        self._index_name = name

        self._index = FAISS.load_local(folder_path=self._parent._local_dir,
                                       embeddings=self._parent._embeddings,
                                       index_name=self._index_name,
                                       allow_dangerous_deserialization=True)

    def insert(self, data: list[list] | list[dict], **kwargs) -> dict:
        """
        Insert data into the vector database.

        Parameters
        ----------
        data : list[list] | list[dict]
            Data to be inserted into the collection.
        **kwargs
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns response content as a dictionary.
        """
        raise NotImplementedError("Insert operation is not supported in FAISS")

    def insert_dataframe(self, df: DataFrameType, **kwargs) -> dict:
        """
        Insert a dataframe entires into the vector database.

        Parameters
        ----------
        df : DataFrameType
            Dataframe to be inserted into the collection.
        **kwargs
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns response content as a dictionary.
        """
        raise NotImplementedError("Insert operation is not supported in FAISS")

    def describe(self, **kwargs) -> dict:
        """
        Provides a description of the collection.

        Parameters
        ----------
        **kwargs
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns response content as a dictionary.
        """
        return {
            "index_name": self._index_name,
            "folder_path": self._folder_path,
        }

    def query(self, query: str, **kwargs) -> typing.Any:
        """
        Query data in a collection in the vector database.

        Parameters
        ----------
        query : str, optional
            The search query, which can be a filter expression, by default None.
        **kwargs
            Additional keyword arguments for the search operation.

        Returns
        -------
        typing.Any
            The search result, which can vary depending on the query and options.
        """
        raise NotImplementedError("Query operation is not supported in FAISS")

    async def similarity_search(self, embeddings: list[list[float]], k: int = 4, **kwargs) -> list[list[dict]]:
        """
        Perform a similarity search within the FAISS docstore.

        Parameters
        ----------
        embeddings : list[list[float]]
            Embeddings for which to perform the similarity search.
        k : int, optional
            The number of nearest neighbors to return, by default 4.
        **kwargs
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        list[dict]
            Returns a list of dictionaries representing the results of the similarity search.
        """

        async def single_search(single_embedding):
            docs = await self._index.asimilarity_search_by_vector(embedding=single_embedding, k=k)

            return [d.dict() for d in docs]

        return await asyncio.gather(*(single_search(embedding) for embedding in embeddings))

    def update(self, data: list[typing.Any], **kwargs) -> dict[str, typing.Any]:
        """
        Update data in the collection.

        Parameters
        ----------
        data : list[typing.Any]
            Data to be updated in the collection.
        **kwargs
            Extra keyword arguments specific to upsert operation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the updated operation stats.
        """
        raise NotImplementedError("Update operation is not supported in FAISS")

    def delete_by_keys(self, keys: int | str | list, **kwargs) -> typing.Any:
        """
        Delete vectors by keys from the collection.

        Parameters
        ----------
        keys : int | str | list
            Primary keys to delete vectors.
        **kwargs
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns result of the given keys that are deleted from the collection.
        """
        raise NotImplementedError("Delete by keys operation is not supported in FAISS")

    def delete(self, expr: str, **kwargs) -> dict[str, typing.Any]:
        """
        Delete vectors by giving a list of IDs.

        Parameters
        ----------
        expr : str
            Delete expression.
        **kwargs
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the given keys that are deleted from the collection.
        """
        raise NotImplementedError("delete operation is not supported in FAISS")

    def retrieve_by_keys(self, keys: int | str | list, **kwargs) -> list[typing.Any]:
        """
        Retrieve the inserted vectors using their primary keys.

        Parameters
        ----------
        keys : int | str | list
            Primary keys to get vectors for. Depending on pk_field type it can be int or str
            or a list of either.
        **kwargs
            Additional keyword arguments for the retrieval operation.

        Returns
        -------
        list[typing.Any]
            Returns result rows of the given keys from the collection.
        """
        raise NotImplementedError("Retrieve by keys operation is not supported in FAISS")

    def count(self, **kwargs) -> int:
        """
        Returns number of rows/entities.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for the count operation.

        Returns
        -------
        int
            Returns number of entities in the collection.
        """
        return self._index.index.ntotal

    def drop(self, **kwargs) -> None:
        """
        Drops the resource from the vector database service.

        This function allows you to drop a collection.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments for specifying the type and partition name (if applicable).
        """
        raise NotImplementedError("Drop operation is not supported in FAISS")


class FaissVectorDBService(VectorDBService):
    """
    Service class for FAISS Vector Database implementation. This class provides functions for interacting
    with a FAISS vector database.

    Parameters
    ----------
    local_dir : str
        The local directory where the FAISS index files are stored.
    embeddings : Embeddings
        The embeddings object to use for embedding text.
    """

    _collection_locks = {}
    _cleanup_interval = 600  # 10mins
    _last_cleanup_time = time.time()

    def __init__(self, local_dir: str, embeddings: "Embeddings"):

        if IMPORT_EXCEPTION is not None:
            raise ImportError(IMPORT_ERROR_MESSAGE.format(package='langchain and faiss-gpu')) from IMPORT_EXCEPTION

        self._local_dir = local_dir
        self._embeddings = embeddings

    @property
    def embeddings(self):
        return self._embeddings

    def load_resource(self, name: str = "index", **kwargs) -> FaissVectorDBResourceService:
        """
        Loads a VDB resource into memory for use.

        Parameters
        ----------
        name : str, optional
            The VDB resource to load. For FAISS, this corresponds to the index name, by default "index"
        **kwargs
            Additional keyword arguments specific to the resource service.

        Returns
        -------
        FaissVectorDBResourceService
            The loaded resource service.
        """

        return FaissVectorDBResourceService(self, name=name, **kwargs)

    def has_store_object(self, name: str) -> bool:
        """
        Check if specific index file name exists by attempting to load FAISS index, docstore,
        and index_to_docstore_id from disk with the index file name.

        Parameters
        ----------
        name : str
            Name of the FAISS index file to check.

        Returns
        -------
        bool
            True if the file exists, False otherwise.
        """
        try:
            FAISS.load_local(folder_path=self._local_dir,
                             embeddings=self._embeddings,
                             index_name=name,
                             allow_dangerous_deserialization=True)
            return True
        except Exception:
            return False

    def list_store_objects(self, **kwargs) -> list[str]:
        """
        List the names of all resources in the vector database.

        Returns
        -------
        list[str]
            A list of collection names.
        """
        raise NotImplementedError("list_store_objects operation is not supported in FAISS")

    def create(self, name: str, overwrite: bool = False, **kwargs):
        """
        Create a collection.

        Parameters
        ----------
        name : str
            Name of the collection to be created.
        overwrite : bool, optional
            If True, the collection will be overwritten if it already exists, by default False.
        **kwargs
            Additional keyword arguments containing collection configuration.

        Raises
        ------
        ValueError
            If the provided schema fields configuration is empty.
        """
        raise NotImplementedError("create operation is not supported in FAISS")

    def create_from_dataframe(self, name: str, df: DataFrameType, overwrite: bool = False, **kwargs) -> None:
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
        **kwargs
            Extra keyword arguments specific to the vector database implementation.
        """

        raise NotImplementedError("create_from_dataframe operation is not supported in FAISS")

    def insert(self, name: str, data: list[list] | list[dict], **kwargs) -> dict[str, typing.Any]:
        """
        Insert a collection specific data in the vector database.

        Parameters
        ----------
        name : str
            Name of the collection to be inserted.
        data : list[list] | list[dict]
            Data to be inserted in the collection.
        **kwargs
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

        raise NotImplementedError("create_from_dataframe operation is not supported in FAISS")

    def insert_dataframe(self, name: str, df: DataFrameType, **kwargs) -> dict[str, typing.Any]:
        """
        Converts dataframe to rows and insert to the vector database.

        Parameters
        ----------
        name : str
            Name of the collection to be inserted.
        df : DataFrameType
            Dataframe to be inserted in the collection.
        **kwargs
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
        raise NotImplementedError("insert_dataframe operation is not supported in FAISS")

    def query(self, name: str, query: str = None, **kwargs) -> typing.Any:
        """
        Query data in a vector database.

        Parameters
        ----------
        name : str
            Name of the collection to search within.
        query : str
            The search query, which can be a filter expression.
        **kwargs
            Additional keyword arguments for the search operation.

        Returns
        -------
        typing.Any
            The search result, which can vary depending on the query and options.
        """

        raise NotImplementedError("query operation is not supported in FAISS")

    async def similarity_search(self, name: str, **kwargs) -> list[dict]:
        """
        Perform a similarity search within the collection.

        Parameters
        ----------
        name : str
            Name of the collection.
        **kwargs
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        list[dict]
            Returns a list of dictionaries representing the results of the similarity search.
        """

        raise NotImplementedError("similarity_search operation is not supported in FAISS")

    def update(self, name: str, data: list[typing.Any], **kwargs) -> dict[str, typing.Any]:
        """
        Update data in the vector database.

        Parameters
        ----------
        name : str
            Name of the collection.
        data : list[typing.Any]
            Data to be updated in the collection.
        **kwargs
            Extra keyword arguments specific to upsert operation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the updated operation stats.
        """

        raise NotImplementedError("update operation is not supported in FAISS")

    def delete_by_keys(self, name: str, keys: int | str | list, **kwargs) -> typing.Any:
        """
        Delete vectors by keys from the collection.

        Parameters
        ----------
        name : str
            Name of the collection.
        keys : int | str | list
            Primary keys to delete vectors.
        **kwargs
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns result of the given keys that are delete from the collection.
        """

        raise NotImplementedError("delete_by_keys operation is not supported in FAISS")

    def delete(self, name: str, expr: str, **kwargs) -> dict[str, typing.Any]:
        """
        Delete vectors from the collection using expressions.

        Parameters
        ----------
        name : str
            Name of the collection.
        expr : str
            Delete expression.
        **kwargs
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the given keys that are delete from the collection.
        """

        raise NotImplementedError("delete operation is not supported in FAISS")

    def retrieve_by_keys(self, name: str, keys: int | str | list, **kwargs) -> list[typing.Any]:
        """
        Retrieve the inserted vectors using their primary keys from the Collection.

        Parameters
        ----------
        name : str
            Name of the collection.
        keys : int | str | list
            Primary keys to get vectors for. Depending on pk_field type it can be int or str
            or a list of either.
        **kwargs
            Additional keyword arguments for the retrieval operation.

        Returns
        -------
        list[typing.Any]
            Returns result rows of the given keys from the collection.
        """

        raise NotImplementedError("retrieve_by_keys operation is not supported in FAISS")

    def count(self, name: str, **kwargs) -> int:
        """
        Returns number of rows/entities in the given collection.

        Parameters
        ----------
        name : str
            Name of the collection.
        **kwargs
            Additional keyword arguments for the count operation.

        Returns
        -------
        int
            Returns number of entities in the collection.
        """

        raise NotImplementedError("count operation is not supported in FAISS")

    def drop(self, name: str, **kwargs) -> None:
        """
        Drop a collection.

        Parameters
        ----------
        name : str
            Name of the collection, index, or partition to be dropped.
        **kwargs
            Additional keyword arguments for specifying the type and partition name (if applicable).

        Raises
        ------
        ValueError
            If mandatory arguments are missing or if the provided 'collection' value is invalid.
        """

        raise NotImplementedError("drop operation is not supported in FAISS")

    def describe(self, name: str, **kwargs) -> dict:
        """
        Describe the collection in the vector database.

        Parameters
        ----------
        name : str
            Name of the collection.
        **kwargs
            Additional keyword arguments specific to the vector database.

        Returns
        -------
        dict
            Returns collection information.
        """

        raise NotImplementedError("describe operation is not supported in FAISS")

    def release_resource(self, name: str) -> None:
        """
        Release a loaded collection from the memory.

        Parameters
        ----------
        name : str
            Name of the collection to release.
        """

        raise NotImplementedError("release_resource operation is not supported in FAISS")

    def close(self) -> None:
        """
        Close the vector database service and release all resources.
        """
        raise NotImplementedError("close operation is not supported in FAISS")
