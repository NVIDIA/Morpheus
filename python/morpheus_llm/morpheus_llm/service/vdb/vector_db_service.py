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

import logging
import typing
from abc import ABC
from abc import abstractmethod

from morpheus.utils.type_aliases import DataFrameType

logger = logging.getLogger(__name__)


class VectorDBResourceService(ABC):
    """
    Abstract base class for a Vector Database Resource Service.
    """

    @abstractmethod
    def insert(self, data: list[list] | list[dict], **kwargs: dict[str, typing.Any]) -> dict:
        """
        Insert data into the vector database.

        Parameters
        ----------
        data : list[list] | list[dict]
            Data to be inserted into the resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns response content as a dictionary.
        """

        pass

    @abstractmethod
    def insert_dataframe(self, df: DataFrameType, **kwargs: dict[str, typing.Any]) -> dict:
        """
        Insert a dataframe into the vector database.

        Parameters
        ----------
        df : DataFrameType
            Dataframe to be inserted into the resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns response content as a dictionary.
        """

        pass

    @abstractmethod
    def describe(self, **kwargs: dict[str, typing.Any]) -> dict:
        """
        Provide a description of the vector database.

        Parameters
        ----------
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns response content as a dictionary.
        """

        pass

    @abstractmethod
    def update(self, data: list[typing.Any], **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Update data in the vector database.

        Parameters
        ----------
        data : list[typing.Any]
            Data to be updated in the resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the updated operation stats.
        """

        pass

    @abstractmethod
    def delete(self, expr: str, **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Delete data in the vector database.

        Parameters
        ----------
        expr : typing.Any
            Delete expression.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the delete operation stats.
        """

        pass

    @abstractmethod
    def retrieve_by_keys(self, keys: int | str | list, **kwargs: dict[str, typing.Any]) -> list[typing.Any]:
        """
        Retrieve the inserted vectors using keys from the resource.

        Parameters
        ----------
        keys : typing.Any
            Primary keys to get vectors.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        list[typing.Any]
            Returns rows of the given keys that exists in the resource.
        """
        pass

    @abstractmethod
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

    @abstractmethod
    def count(self, **kwargs: dict[str, typing.Any]) -> int:
        """
        Returns number of rows/entities in the given resource.

        Parameters
        ----------
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        int
            Returns number of rows/entities in the given resource.
        """
        pass

    @abstractmethod
    async def similarity_search(self,
                                embeddings: list[list[float]],
                                k: int = 4,
                                **kwargs: dict[str, typing.Any]) -> list[list[dict]]:
        """
        Perform a similarity search within the vector database.

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
            Returns a list of lists, where each inner list contains dictionaries representing the results of the
            similarity search.
        """

        pass


class VectorDBService(ABC):
    """
    Class used for vectorstore specific implementation.
    """

    @abstractmethod
    def load_resource(self, name: str, **kwargs: dict[str, typing.Any]) -> VectorDBResourceService:
        pass

    @abstractmethod
    def insert(self, name: str, data: list[list] | list[dict], **kwargs: dict[str, typing.Any]) -> dict:
        """
        Insert data into the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        data : list[list] | list[dict]
            Data to be inserted into the resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns response content as a dictionary.
        """

        pass

    @abstractmethod
    def insert_dataframe(self, name: str, df: DataFrameType, **kwargs: dict[str, typing.Any]) -> dict:
        """
        Converts dataframe to rows and insert into the vector database resource.

        Parameters
        ----------
        name : str
            Name of the resource to be inserted.
        df : DataFrameType
            Dataframe to be inserted.
        **kwargs : dict[str, typing.Any]
            Additional keyword arguments containing collection configuration.

        Returns
        -------
        dict
            Returns response content as a dictionary.

        Raises
        ------
        RuntimeError
            If the resource not exists exists.
        """
        pass

    @abstractmethod
    def query(self, name: str, query: str, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Query a resource in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        query : str
            Query to execute on the given resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns search results.
        """

        pass

    @abstractmethod
    async def similarity_search(self, name: str, **kwargs: dict[str, typing.Any]) -> list[list[dict]]:
        """
        Perform a similarity search within the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        list[list[dict]]
            Returns a list of lists, where each inner list contains dictionaries representing the results of the
            similarity search.
        """

        pass

    @abstractmethod
    def drop(self, name: str, **kwargs: dict[str, typing.Any]) -> None:
        """
        Drop resources from the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.
        """

        pass

    @abstractmethod
    def update(self, name: str, data: list[typing.Any], **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Update data in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        data : list[typing.Any]
            Data to be updated in the resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the updated operation stats.
        """

        pass

    @abstractmethod
    def delete(self, name: str, expr: str, **kwargs: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """
        Delete data in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        expr : typing.Any
            Delete expression.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict[str, typing.Any]
            Returns result of the delete operation stats.
        """

        pass

    @abstractmethod
    def create(self, name: str, overwrite: bool = False, **kwargs: dict[str, typing.Any]) -> None:
        """
        Create resources in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        overwrite : bool, default False
            Whether to overwrite the resource if it already exists.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.
        """

        pass

    @abstractmethod
    def create_from_dataframe(self,
                              name: str,
                              df: DataFrameType,
                              overwrite: bool = False,
                              **kwargs: dict[str, typing.Any]) -> None:
        """
        Create resources in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        df : DataFrameType
            The dataframe to create the resource from.
        overwrite : bool, optional
            Whether to overwrite the resource if it already exists. Default is False.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.
        """
        pass

    @abstractmethod
    def describe(self, name: str, **kwargs: dict[str, typing.Any]) -> dict:
        """
        Describe resource in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        dict
            Returns resource information.
        """

        pass

    @abstractmethod
    def release_resource(self, name: str) -> None:
        """
        Release a loaded resource from the memory.

        Parameters
        ----------
        name : str
            Name of the resource to release.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close connection to the vector database.
        """

        pass

    @abstractmethod
    def has_store_object(self, name: str) -> bool:
        """
        Check if a resource exists in the vector database.

        Parameters
        ----------
        name : str
            Name of the resource.

        Returns
        -------
        bool
            Returns True if resource exists in the vector database, otherwise False.
        """

        pass

    @abstractmethod
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

    # pylint: disable=unused-argument
    def transform(self, data: typing.Any, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Transform data according to the specific vector database implementation.

        Parameters
        ----------
        data : typing.Any
            Data to be updated in the resource.
        **kwargs : dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        typing.Any
            Returns transformed data as per the implementation.
        """
        return data

    @abstractmethod
    def retrieve_by_keys(self, name: str, keys: int | str | list, **kwargs: dict[str, typing.Any]) -> list[typing.Any]:
        """
        Retrieve the inserted vectors using keys from the resource.

        Parameters
        ----------
        name : str
            Name of the resource.
        keys : typing.Any
            Primary keys to get vectors.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        list[typing.Any]
            Returns rows of the given keys that exists in the resource.
        """
        pass

    @abstractmethod
    def delete_by_keys(self, name: str, keys: int | str | list, **kwargs: dict[str, typing.Any]) -> typing.Any:
        """
        Delete vectors by keys from the resource.

        Parameters
        ----------
        name : str
            Name of the resource.
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

    @abstractmethod
    def count(self, name: str, **kwargs: dict[str, typing.Any]) -> int:
        """
        Returns number of rows/entities in the given resource.

        Parameters
        ----------
        name : str
            Name of the resource.
        **kwargs :  dict[str, typing.Any]
            Extra keyword arguments specific to the vector database implementation.

        Returns
        -------
        int
            Returns number of rows/entities in the given resource.
        """
        pass
