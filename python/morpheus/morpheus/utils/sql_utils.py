# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
from sqlite3 import Row

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import exc
from sqlalchemy.engine.url import URL
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(f"morpheus.{__name__}")


class SQLUtils:
    """
    Utility class for working with SQL databases using SQLAlchemy.

    Parameters
    ----------
    sql_config : typing.Dict[any, any]
        Configuration parameters for the SQL connection.
    pool_size : int
        The number of connections to keep in the connection pool (default: 5).
    max_overflow : int
        The maximum number of connections that can be created beyond the pool_size (default: 5).

    """

    def __init__(self, sql_config: typing.Dict[any, any], pool_size: int = 5, max_overflow: int = 5):
        self._dialect = sql_config["drivername"]
        self._engine = None
        self._sql_config = sql_config
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._connection_string = self.gen_sql_conn_str(sql_config)

    @property
    def connection_string(self):
        return self._connection_string

    def connect(self) -> None:
        """
        Establishes a connection to the database using the specified SQL configuration.

        """
        self._engine = create_engine(
            self._connection_string,
            poolclass=QueuePool,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
        )

    def close(self) -> None:
        """
        Closes the database connection.

        """
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def execute_query(self, query: str, params: typing.Tuple = ()) -> None:
        """
        Executes a SQL query without returning any result.

        Parameters
        ----------
        query : str
            The SQL query to execute.
        params : typing.Tuple, optional
            The query parameters (default: ()).

        """
        try:
            with self._engine.connect() as connection:
                connection.execute(query, params)
        except exc.SQLAlchemyError:
            logger.exception("Error executing query: %s", query)
            raise

    def execute_query_with_result(self, query: str, params: typing.Tuple = ()) -> typing.List[Row]:
        """
        Executes a SQL query and returns the result as a list of rows.

        Parameters
        ----------
        query : str
            The SQL query to execute.
        params : typing.Tuple, optional
            The query parameters (default: ()).

        Returns
        -------
        rows : typing.List[Row]
            The result of the query as a list of rows.

        """
        try:
            with self._engine.connect() as connection:
                result = connection.execute(query, params)
                return result.fetchall()
        except exc.SQLAlchemyError:
            logger.exception("Error executing query: %s", query)
            raise

    def execute_queries_in_transaction(self, queries: typing.List[str], params: typing.Tuple = ()) -> None:
        """
        Executes multiple SQL queries within a transaction.

        Parameters
        ----------
        queries : typing.List[str]
            The list of SQL queries to execute.
        params : typing.Tuple, optional
            The query parameters (default: ()).

        """
        with self._engine.begin() as connection:
            try:
                for query in queries:
                    logger.debug("Executing query: %s", query)
                    connection.execute(query, params)
            except exc.DatabaseError as e:
                error_info = f"Error executing queries in transaction: {str(e)}"
                logger.exception(error_info)
                raise

    def create_table(self, table_name: str, columns: str) -> None:
        """
        Creates a table in the database.

        Parameters
        ----------
        table_name : str
            The name of the table to create.
        columns : str
            The column definitions for the table.

        """
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
        self.execute_query(query)

    def drop_table(self, table_name: str) -> None:
        """
        Drops a table from the database if it exists.

        Parameters
        ----------
        table_name : str
            The name of the table to drop.

        """
        query = f"DROP TABLE IF EXISTS {table_name}"
        self.execute_query(query)

    def insert_data(self, table_name: str, values: typing.Tuple, columns: typing.List[str] = None) -> None:
        """
        Inserts data into a table.

        Parameters
        ----------
        table_name : str
            The name of the table to insert data into.
        values : typing.Tuple
            The values to insert.
        columns : typing.List[str], optional
            The column names to insert data into (default: None).

        """
        num_values = len(values)

        if columns:
            num_columns = len(columns.split(","))
            if num_values != num_columns:
                raise ValueError("Number of column names does not match the number of values")

            query = f"INSERT INTO {table_name} ({columns}) VALUES ({self.gen_placeholder_str(num_columns)})"
            self.execute_query(query, values)
        else:
            query = f"INSERT INTO {table_name} VALUES ({self.gen_placeholder_str(num_values)})"
            self.execute_query(query, values)

    @classmethod
    def gen_sql_conn_str(cls, sql_config: typing.Dict) -> str:
        """
        Generates the SQL connection string based on the SQL configuration.

        Parameters
        ----------
        sql_config : typing.Dict
            Configuration parameters for the SQL connection.

        Returns
        -------
        connection_string : str
            The generated SQL connection string.

        Raises
        ------
        RuntimeError
            If required SQL parameters are missing or invalid.

        """

        possible_params = ["database", "drivername", "host", "port", "username", "password"]

        sql_params = {key: sql_config[key] for key in possible_params if sql_config.get(key)}

        if "drivername" not in sql_params:
            raise RuntimeError("Must specify SQL drivername, ex. 'sqlite'")

        if "host" not in sql_params and "database" not in sql_params:
            raise RuntimeError("Must specify 'host' or 'database'")

        if ("username" in sql_params or "password" in sql_params) and ("username" not in sql_params
                                                                       or "password" not in sql_params):
            raise RuntimeError("Must include both 'username' and 'password' or neither")

        url_obj = URL.create(
            sql_params["drivername"],
            username=sql_params.get("username", None),
            password=sql_params.get("password", None),
            host=sql_params.get("host", None),
            port=sql_params.get("port", None),
            database=sql_params.get("database", None),
        )

        return url_obj.render_as_string(hide_password=False)

    def gen_placeholder_str(self, count: int) -> str:
        """
        Generates the placeholder string for SQL query parameters.

        Parameters
        ----------
        count : int
            The number of placeholders to generate.

        Returns
        -------
        placeholders : str
            The generated placeholder string.

        Raises
        ------
        RuntimeError
            If the count is invalid.

        """
        if count < 1 or count > 1000:
            raise RuntimeError("Invalid number of placeholders")

        if self._dialect == "sqlite":
            placeholders = ', '.join(['?'] * count)
        else:
            placeholders = ', '.join(['%s'] * count)

        return placeholders

    def get_existing_column_rows(self, table_name: str) -> typing.List[str]:
        """
        Retrieves the existing column names of a table.

        Parameters
        ----------
        table_name : str
            The name of the table.

        Returns
        -------
        existing_columns : typing.List[str]
            The existing column names.

        Raises
        ------
        exc.SQLAlchemyError
            If an error occurs while retrieving the existing columns.

        """
        try:
            if self._dialect == "sqlite":
                col_exists_query = f"PRAGMA table_info({table_name})"
                rows = self.execute_query_with_result(query=col_exists_query)
                existing_columns = [column[1] for column in rows]
            else:
                col_exists_query = f"SELECT column_name FROM information_schema.columns WHERE table_name='{table_name}'"
                rows = self.execute_query_with_result(query=col_exists_query)
                existing_columns = [column[0] for column in rows]

            return existing_columns
        except exc.SQLAlchemyError:
            logger.exception("Error retrieving existing columns for table: %s", table_name)
            raise

    def add_columns_if_not_exists(self,
                                  table_name: str,
                                  columns: typing.List[str],
                                  data_type: str,
                                  default_value: typing.Union[int, str] = None) -> bool:
        """
        Adds columns to a table if they don't already exist.

        Parameters
        ----------
        table_name : str
            The name of the table.
        columns : typing.List[str]
            The column names to add.
        data_type : str
            The data type of the columns.
        default_value : typing.Union[int, str], optional
            The default value for the columns (default: None).

        Returns
        -------
        new_col_added : bool
            True if columns were added, False otherwise.

        """
        existing_columns = self.get_existing_column_rows(table_name)
        columns_to_add = [column for column in columns if column not in existing_columns]

        new_col_added = False

        for column in columns_to_add:
            alter_stmt = f"ALTER TABLE {table_name} ADD COLUMN {column} {data_type}"
            if default_value:
                alter_stmt = f"{alter_stmt} DEFAULT {default_value}"
            self.execute_query(query=alter_stmt)
            new_col_added = True
            logger.debug("Statement executed successfully")

        return new_col_added

    def get_table_columns(self, table_name: str) -> typing.List[str]:
        """
        Retrieves the column names of a table.

        Parameters
        ----------
        table_name : str
            The name of the table.

        Returns
        -------
        columns : typing.List[str]
            The column names.

        Raises
        ------
        exc.SQLAlchemyError
            If an error occurs while retrieving the columns.

        """
        try:
            columns = self.get_existing_column_rows(table_name)
            return columns
        except exc.SQLAlchemyError:
            logger.exception("Error retrieving columns for table: %s", table_name)
            raise

    def to_dataframe(self, query: str, params: typing.Tuple = None) -> pd.DataFrame:
        """
        Executes a SQL query and returns the result as a Pandas DataFrame.

        Parameters
        ----------
        query : str
            The SQL query to execute.
        params : typing.Tuple, optional
            The query parameters (default: None).

        Returns
        -------
        df : pd.DataFrame
            The result of the query as a Pandas DataFrame.

        Raises
        ------
        exc.SQLAlchemyError
            If an error occurs while executing the query or converting the result to a DataFrame.

        """
        try:
            with self._engine.connect() as connection:
                df = pd.read_sql(query, connection, params=params)
                return df
        except exc.SQLAlchemyError:
            logger.exception("Error executing query and converting result to dataframe: %s", query)
            raise

    def to_table(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace", index: bool = False) -> None:
        """
        Converts a Pandas DataFrame to a SQL table.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to convert to a table.
        table_name : str
            The name of the table.
        if_exists : str, optional
            Action to take if the table already exists (default: "replace").
        index : bool, optional
            Whether to include the DataFrame index as a column in the table (default: False).

        Raises
        ------
        exc.SQLAlchemyError
            If an error occurs while converting the DataFrame to a table.

        """
        try:
            with self._engine.connect() as connection:
                df.to_sql(table_name, connection, if_exists=if_exists, index=index)
        except exc.SQLAlchemyError:
            logger.exception("Error converting dataframe to table: %s", table_name)
            raise
