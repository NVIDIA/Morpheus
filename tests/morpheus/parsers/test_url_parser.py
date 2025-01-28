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

import pytest

from cudf import DataFrame

from morpheus.parsers import url_parser

# pylint: disable=redefined-outer-name


@pytest.fixture
def input_df():
    return DataFrame({
        "url": [
            "http://www.google.com",
            "gmail.com",
            "github.com",
            "https://pandas.pydata.org",
            "http://www.worldbank.org.kg/",
            "waiterrant.blogspot.com",
            "http://forums.news.cnn.com.ac/",
            "http://forums.news.cnn.ac/",
            "ftp://b.cnn.com/",
            "a.news.uk",
            "a.news.co.uk",
            "https://a.news.co.uk",
            "107-193-100-2.lightspeed.cicril.sbcglobal.net",
            "a23-44-13-2.deploy.static.akamaitechnologies.com",
        ]
    })


def test_parse_1(input_df):
    expected_output_df = DataFrame({
        "domain": [
            "google",
            "gmail",
            "github",
            "pydata",
            "worldbank",
            "waiterrant",
            "cnn",
            "cnn",
            "cnn",
            "news",
            "news",
            "news",
            "sbcglobal",
            "akamaitechnologies",
        ],
        "suffix": [
            "com",
            "com",
            "com",
            "org",
            "org.kg",
            "blogspot.com",
            "com.ac",
            "ac",
            "com",
            "uk",
            "co.uk",
            "co.uk",
            "net",
            "com",
        ],
    })
    output_df = url_parser.parse(input_df["url"], req_cols={"domain", "suffix"})

    assert expected_output_df.equals(output_df)


def test_parse_2(input_df):
    expected_output_df = DataFrame({
        "hostname": [
            "www.google.com",
            "gmail.com",
            "github.com",
            "pandas.pydata.org",
            "www.worldbank.org.kg",
            "waiterrant.blogspot.com",
            "forums.news.cnn.com.ac",
            "forums.news.cnn.ac",
            "b.cnn.com",
            "a.news.uk",
            "a.news.co.uk",
            "a.news.co.uk",
            "107-193-100-2.lightspeed.cicril.sbcglobal.net",
            "a23-44-13-2.deploy.static.akamaitechnologies.com",
        ],
        "subdomain": [
            "www",
            "",
            "",
            "pandas",
            "www",
            "",
            "forums.news",
            "forums.news",
            "b",
            "a",
            "a",
            "a",
            "107-193-100-2.lightspeed.cicril",
            "a23-44-13-2.deploy.static",
        ],
        "domain": [
            "google",
            "gmail",
            "github",
            "pydata",
            "worldbank",
            "waiterrant",
            "cnn",
            "cnn",
            "cnn",
            "news",
            "news",
            "news",
            "sbcglobal",
            "akamaitechnologies",
        ],
        "suffix": [
            "com",
            "com",
            "com",
            "org",
            "org.kg",
            "blogspot.com",
            "com.ac",
            "ac",
            "com",
            "uk",
            "co.uk",
            "co.uk",
            "net",
            "com",
        ],
    })
    output_df = url_parser.parse(input_df["url"])

    assert expected_output_df.equals(output_df)


def test_parse_invalid_req_cols(input_df):
    expected_error = ValueError(
        "Given req_cols must be subset of [\"hostname\", \"subdomain\", \"domain\", \"suffix\"]")
    with pytest.raises(ValueError) as actual_error:
        url_parser.parse(input_df["url"], req_cols={"test"})
        assert actual_error == expected_error
