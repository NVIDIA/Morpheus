/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "./common.hpp"

#include "morpheus/io/data_loader_registry.hpp"
#include "morpheus/io/loaders/file.hpp"
#include "morpheus/io/loaders/grpc.hpp"
#include "morpheus/io/loaders/payload.hpp"
#include "morpheus/io/loaders/rest.hpp"
#include "morpheus/messages/meta.hpp"
#include "morpheus/utilities/string_util.hpp"

#include <cpython/initconfig.h>  // for PyStatus_Exception, PyConfig_Clear, PyConfig_InitPythonConfig
#include <nlohmann/json.hpp>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pylifecycle.h>  // for Py_InitializeFromConfig

#include <cassert>
#include <codecvt>  // for codecvt_utf8_utf16
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifndef PYTHON_EXECUTABLE
    #error PYTHON_EXECUTABLE must be defined to run tests
#endif

namespace morpheus::test {

bool TestWithPythonInterpreter::m_initialized = false;

void TestWithPythonInterpreter::SetUp()
{
    this->initialize_interpreter();

    LoaderRegistry::register_factory_fn(
        "file",
        [](nlohmann::json config) {
            return std::make_unique<FileDataLoader>(config);
        },
        false);
    LoaderRegistry::register_factory_fn(
        "grpc",
        [](nlohmann::json config) {
            return std::make_unique<GRPCDataLoader>(config);
        },
        false);
    LoaderRegistry::register_factory_fn(
        "payload",
        [](nlohmann::json config) {
            return std::make_unique<PayloadDataLoader>(config);
        },
        false);
    LoaderRegistry::register_factory_fn(
        "rest",
        [](nlohmann::json config) {
            return std::make_unique<RESTDataLoader>(config);
        },
        false);

    pybind11::gil_scoped_acquire gil;
}

void TestWithPythonInterpreter::TearDown() {}

void TestWithPythonInterpreter::initialize_interpreter() const
{
    if (!m_initialized)
    {
        using namespace std::string_literals;

        // NOTE: We manually initialize the Python interpreter here because we need to specify the Python executable to
        // use in order to enable virtual environments. Otherwise, the Python interpreter will be initialized with the
        // default executable, which may not be the one we want to use (and will make it difficult to discover why tests
        // are failing).
        PyConfig config;
        PyConfig_InitPythonConfig(&config);

        // Create a wstring from the PYTHON_EXECUTABLE string
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;

        auto python_exe_w = converter.from_bytes(PYTHON_EXECUTABLE);

        // Set the program name to the python executable to ensure any virtualenvs are loaded correctly
        PyStatus status = PyConfig_SetString(&config, &config.program_name, python_exe_w.data());
        if (PyStatus_Exception(status))
        {
            throw std::runtime_error("Failed to set Python program name to "s + PYTHON_EXECUTABLE);
        }

        // Load the remainder of the configuration
        status = PyConfig_Read(&config);
        if (PyStatus_Exception(status))
        {
            throw std::runtime_error("Failed to read Python configuration");
        }

        status = Py_InitializeFromConfig(&config);
        if (PyStatus_Exception(status))
        {
            throw std::runtime_error("Failed to initialize Python interpreter");
        }

        // Cleanup the configuration object
        PyConfig_Clear(&config);

        m_initialized = true;
    }
}

std::filesystem::path get_morpheus_root()
{
    auto root = std::getenv("MORPHEUS_ROOT");

    if (root == nullptr)
    {
        throw std::runtime_error("MORPHEUS_ROOT env variable is not set");
    }

    return std::filesystem::path{root};
}

std::string create_mock_csv_file(std::vector<std::string> cols, std::vector<std::string> dtypes, std::size_t rows)
{
    assert(cols.size() == dtypes.size());
    static std::vector<std::string> random_strings = {"field1", "test123", "abc", "xyz", "123", "foo", "bar", "baz"};

    auto sstream = std::stringstream();

    // Create header
    sstream << StringUtil::join(cols.begin(), cols.end(), ",");
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
                throw std::runtime_error(dtypes[col] + ": No");
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
    auto string_df = create_mock_csv_file(cols, dtypes, rows);

    pybind11::gil_scoped_acquire gil;
    pybind11::module_ mod_cudf;
    mod_cudf = pybind11::module_::import("cudf");

    auto py_string = pybind11::str(string_df);
    auto py_buffer = pybind11::buffer(pybind11::bytes(py_string));
    auto dataframe = mod_cudf.attr("read_csv")(py_buffer);

    return MessageMeta::create_from_python(std::move(dataframe));
}

}  // namespace morpheus::test
