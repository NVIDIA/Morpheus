/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../test_utils/common.hpp"  // IWYU pragma: associated
#include "test_io.hpp"

#include "morpheus/io/data_loader_registry.hpp"
#include "morpheus/io/loaders/payload.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <pybind11/embed.h>

#include <memory>
#include <stdexcept>

namespace py = pybind11;
using namespace morpheus;
using namespace morpheus::test;

TEST_F(TestDataLoaderRegistry, LoaderRegistryContainsTest)
{
    ASSERT_FALSE(LoaderRegistry::contains("no_a_loader"));

    ASSERT_TRUE(LoaderRegistry::contains("file"));
    ASSERT_TRUE(LoaderRegistry::contains("grpc"));
    ASSERT_TRUE(LoaderRegistry::contains("payload"));
    ASSERT_TRUE(LoaderRegistry::contains("rest"));
}

TEST_F(TestDataLoaderRegistry, LoaderRegistryRegisterLoaderTest)
{
    ASSERT_FALSE(LoaderRegistry::contains("LoaderRegistryRegisterLoaderTest"));

    // Should be able to register a loader
    LoaderRegistry::register_factory_fn("LoaderRegistryRegisterLoaderTest", [](nlohmann::json config) {
        return std::make_unique<PayloadDataLoader>(config);
    });
    ASSERT_TRUE(LoaderRegistry::contains("LoaderRegistryRegisterLoaderTest"));

    // Should be able to overwrite an existing loader if we request it
    EXPECT_NO_THROW(LoaderRegistry::register_factory_fn(
        "LoaderRegistryRegisterLoaderTest",
        [](nlohmann::json config) {
            return std::make_unique<PayloadDataLoader>(config);
        },
        false));

    EXPECT_THROW(LoaderRegistry::register_factory_fn("LoaderRegistryRegisterLoaderTest",
                                                     [](nlohmann::json config) {
                                                         return std::make_unique<PayloadDataLoader>(config);
                                                     }),
                 std::runtime_error);
}

TEST_F(TestDataLoaderRegistry, LoaderRegistryUnregisterLoaderTest)
{
    ASSERT_FALSE(LoaderRegistry::contains("LoaderRegistryUnregisterLoaderTest"));

    // Should be able to register a loader
    LoaderRegistry::register_factory_fn("LoaderRegistryUnregisterLoaderTest", [](nlohmann::json config) {
        return std::make_unique<PayloadDataLoader>(config);
    });
    ASSERT_TRUE(LoaderRegistry::contains("LoaderRegistryUnregisterLoaderTest"));

    // Should be able to unregister a loader
    LoaderRegistry::unregister_factory_fn("LoaderRegistryUnregisterLoaderTest");
    ASSERT_FALSE(LoaderRegistry::contains("LoaderRegistryUnregisterLoaderTest"));

    ASSERT_THROW(LoaderRegistry::unregister_factory_fn("LoaderRegistryUnregisterLoaderTest"), std::runtime_error);
    ASSERT_NO_THROW(LoaderRegistry::unregister_factory_fn("LoaderRegistryUnregisterLoaderTest", false));
}
