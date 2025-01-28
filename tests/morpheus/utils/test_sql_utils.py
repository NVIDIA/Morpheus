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

from unittest.mock import MagicMock

import pytest

from morpheus.utils.sql_utils import SQLUtils


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="function")
def sql_utils(mock_connection):
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_connection

    utils_ = SQLUtils({"drivername": "sqlite", "database": "mydatabase.db"}, pool_size=5, max_overflow=5)
    utils_._engine = mock_engine

    return utils_


@pytest.fixture(scope="function")
def mock_connection():
    return MagicMock()


def test_execute_query(sql_utils):
    query = "SELECT * FROM table"
    sql_utils.execute_query(query)

    # Assert that the connection's execute method is called with the query
    sql_utils._engine.connect.assert_called_once()
    sql_utils._engine.connect.return_value.__enter__.return_value.execute.assert_called_once_with(query, ())


def test_execute_query_with_result(sql_utils, mock_connection):
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [("apple", 50), ("banana", 30)]

    mock_connection.execute.return_value = mock_result

    query = "SELECT * FROM table"
    result = sql_utils.execute_query_with_result(query)

    mock_connection.execute.assert_called_once_with(query, ())

    mock_result.fetchall.assert_called()

    # Assert the returned result matches the expected result
    assert result == [("apple", 50), ("banana", 30)]


def test_create_table(sql_utils, mock_connection):
    table_name = "my_table"
    columns = "id INT, name VARCHAR(50)"
    sql_utils.create_table(table_name, columns)

    # Assert that the connection's execute method is called with the create table query
    create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
    mock_connection.execute.assert_called_once_with(create_query, ())


def test_drop_table(sql_utils, mock_connection):
    table_name = "my_table"
    sql_utils.drop_table(table_name)

    # Assert that the connection's execute method is called with the drop table query
    drop_query = f"DROP TABLE IF EXISTS {table_name}"
    mock_connection.execute.assert_called_once_with(drop_query, ())


def test_insert_data(sql_utils, mock_connection):
    table_name = "my_table"
    values = [1, "John"]
    columns = "id, name"
    sql_utils.insert_data(table_name, values, columns)

    # Assert that the connection's execute method is called with the insert query
    insert_query = f"INSERT INTO {table_name} ({columns}) VALUES (?, ?)"
    mock_connection.execute.assert_called_with(insert_query, values)


@pytest.mark.parametrize(
    "sql_config, expected_conn_str, expected_exception",
    [
        # Valid configurations
        ({
            "drivername": "postgres+psycopg2", "host": "localhost", "database": "mydb"
        },
         "postgres+psycopg2://localhost/mydb",
         None),
        # Valid configurations
        ({
            "drivername": "mysql+pymysql",
            "host": "localhost",
            "port": 3306,
            "database": "mydb",
            "username": "user",
            "password": "password"
        },
         "mysql+pymysql://user:password@localhost:3306/mydb",
         None),
        # Missing drivername
        ({
            "host": "localhost", "database": "mydb"
        }, None, RuntimeError("Must specify SQL drivername, ex. 'sqlite'")),
        # Missing host and database
        ({
            "drivername": "postgres+psycopg2"
        }, None, RuntimeError("Must specify 'host' or 'database'")),
        # Missing username or password
        ({
            "drivername": "mysql+pymysql", "host": "localhost", "port": 3306, "database": "mydb", "username": "user"
        },
         None,
         RuntimeError("Must include both 'username' and 'password' or neither")),
    ])
def test_gen_sql_conn_str(sql_config, expected_conn_str, expected_exception):
    if expected_exception:
        with pytest.raises(type(expected_exception), match=str(expected_exception)):
            SQLUtils.gen_sql_conn_str(sql_config)
    else:
        conn_str = SQLUtils.gen_sql_conn_str(sql_config)
        assert conn_str == expected_conn_str
