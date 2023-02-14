/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "../test_morpheus.hpp"  // IWYU pragma: associated

#include "morpheus/messages/meta.hpp"

#include <pybind11/pybind11.h>
#include <pymrc/utilities/object_cache.hpp>

#include <chrono>
#include <ctime>
#include <memory>
#include <random>
#include <sstream>

namespace morpheus::test {

/**
 * @brief Test fixture for IO tests
 * Note: we don't finalize the interpreter after each test, because cudf doesn't behave well when the interpreter is
 * initialized more than once. This means that additional attention is required when adding new tests to this fixture,
 * because they will share the same interpreter instance and state.
 */
class TestIO : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        if (!m_initialized)
        {
            pybind11::initialize_interpreter();
            m_initialized = true;

            auto& cache_handle = mrc::pymrc::PythonObjectCache::get_handle();
            cache_handle.get_module("cudf");  // pre-load cudf
        }
    }

    void TearDown() override {}

  private:
    static bool m_initialized;
};

std::string accum_merge(std::string lhs, std::string rhs)
{
    if (lhs.empty())
    {
        return std::move(rhs);
    }

    return std::move(lhs) + "," + std::move(rhs);
}

std::string create_mock_dataframe(std::vector<std::string> cols, std::vector<std::string> dtypes, std::size_t rows)
{
    assert(cols.size() == dtypes.size());
    static std::vector<std::string> random_strings = {"field1", "test123", "abc", "xyz", "123", "foo", "bar", "baz"};

    auto sstream = std::stringstream();

    // Create header
    sstream << std::accumulate(cols.begin(), cols.end(), std::string(""), accum_merge);
    sstream << std::endl;

    // Populate with random data
    std::srand(std::time(nullptr));
    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t col = 0; col < cols.size(); ++col)
        {
            if (dtypes[col] == "int32")
            {
                sstream << std::rand() % 100 << ",";
            }
            else if (dtypes[col] == "float32")
            {
                sstream << std::rand() % 100 << "." << std::rand() % 100 << ",";
            }
            else if (dtypes[col] == "string")
            {
                sstream << random_strings[std::rand() % (random_strings.size() - 1)] << ",";
            }
            else
            {
                throw std::runtime_error("Unsupported dtype");
            }
        }
        sstream.seekp(-1, std::ios::cur);  // Remove last comma
        sstream << std::endl;
    }

    return sstream.str();
}

std::shared_ptr<MessageMeta> create_mock_msg_meta(std::vector<std::string> cols,
                                                  std::vector<std::string> dtypes,
                                                  std::size_t rows)
{
    auto string_df = create_mock_dataframe(cols, dtypes, rows);

    pybind11::gil_scoped_acquire gil;
    pybind11::module_ mod_cudf;

    auto& cache_handle = mrc::pymrc::PythonObjectCache::get_handle();
    mod_cudf = cache_handle.get_module("cudf");

    auto py_string = pybind11::str(string_df);
    auto py_buffer = pybind11::buffer(pybind11::bytes(py_string));
    auto dataframe = mod_cudf.attr("read_csv")(py_buffer);

    return MessageMeta::create_from_python(std::move(dataframe));
}

using TestDataLoader = TestIO;  // NOLINT
}  // namespace morpheus::test