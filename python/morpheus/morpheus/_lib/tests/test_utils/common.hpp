/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <glog/logging.h>  // IWYU pragma: keep
#include <gtest/gtest.h>   // IWYU pragma: keep

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#define TEST_CLASS(name)                                                             \
    class __attribute__((visibility("default"))) Test##name : public ::testing::Test \
    {                                                                                \
        void SetUp() override {}                                                     \
    }

#define TEST_CLASS_WITH_PYTHON(name)                                                                           \
    class __attribute__((visibility("default"))) Test##name : public morpheus::test::TestWithPythonInterpreter \
    {}

namespace morpheus {
class MessageMeta;
}

namespace morpheus::test {

/**
 * @brief Test fixture for tests that require a python interpreter.
 * Note: we don't finalize the interpreter after each test, because cudf doesn't behave well when the interpreter is
 * initialized more than once. This means that additional attention is required when adding new tests to this fixture,
 * because they will share the same interpreter instance and state.
 * Note: Additionally, creating another interpreter in the same library (lib_testmorpheus.so) will fail; if you must do
 * so, create a new library.
 */
class TestWithPythonInterpreter : public ::testing::Test
{
  public:
    void initialize_interpreter() const;

  protected:
    void SetUp() override;

    void TearDown() override;

  private:
    static bool m_initialized;
};

/**
 * @brief Gets the `MORPHEUS_ROOT` env variable or throws a runtime_error.
 * @return std::filesystem::path
 */
std::filesystem::path get_morpheus_root();

std::string create_mock_csv_file(std::vector<std::string> cols, std::vector<std::string> dtypes, std::size_t rows);

std::shared_ptr<MessageMeta> create_mock_msg_meta(std::vector<std::string> cols,
                                                  std::vector<std::string> dtypes,
                                                  std::size_t rows);

}  // namespace morpheus::test
