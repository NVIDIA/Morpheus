/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../test_utils/common.hpp"  // for TEST_CLASS_WITH_PYTHON, morpheus

#include "morpheus/controllers/monitor_controller.hpp"  // for MonitorController
#include "morpheus/messages/control.hpp"                // for ControlMessage
#include "morpheus/messages/meta.hpp"                   // for MessageMeta

#include <cuda_runtime.h>                    // for cudaMemcpy, cudaMemcpyKind
#include <cudf/column/column.hpp>            // for column
#include <cudf/column/column_factories.hpp>  // for make_numeric_column
#include <cudf/column/column_view.hpp>       // for mutable_column_view
#include <cudf/io/types.hpp>                 // for column_name_info, table_with_metadata, table_metadata
#include <cudf/table/table.hpp>              // for table
#include <cudf/types.hpp>                    // for type_id, data_type
#include <gtest/gtest.h>                     // for Message, TestPartResult, EXPECT_EQ, TestInfo, EXPECT_...

#include <cstdint>        // for int32_t
#include <functional>     // for function
#include <memory>         // for unique_ptr, shared_ptr, allocator, make_shared, make_...
#include <numeric>        // for iota
#include <optional>       // for optional
#include <stdexcept>      // for runtime_error
#include <unordered_map>  // for unordered_map
#include <utility>        // for move
#include <vector>         // for vector

using namespace morpheus;

TEST_CLASS_WITH_PYTHON(MonitorController);

cudf::io::table_with_metadata create_cudf_table_with_metadata(int rows, int cols)
{
    std::vector<std::unique_ptr<cudf::column>> columns;

    for (int i = 0; i < cols; ++i)
    {
        auto col      = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, rows);
        auto col_view = col->mutable_view();

        std::vector<int32_t> data(rows);
        std::iota(data.begin(), data.end(), 0);
        cudaMemcpy(col_view.data<int32_t>(), data.data(), data.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

        columns.push_back(std::move(col));
    }

    auto table = std::make_unique<cudf::table>(std::move(columns));

    auto index_info   = cudf::io::column_name_info{""};
    auto column_names = std::vector<cudf::io::column_name_info>(cols, index_info);
    auto metadata     = cudf::io::table_metadata{std::move(column_names), {}, {}};

    return cudf::io::table_with_metadata{std::move(table), metadata};
}

TEST_F(TestMonitorController, TestAutoCountFn)
{
    auto message_meta_mc            = MonitorController<std::shared_ptr<MessageMeta>>("test_message_meta");
    auto message_meta_auto_count_fn = message_meta_mc.auto_count_fn();
    auto meta                       = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(10, 2)));
    EXPECT_EQ((*message_meta_auto_count_fn)(meta), 10);

    auto control_message_mc            = MonitorController<std::shared_ptr<ControlMessage>>("test_control_message");
    auto control_message_auto_count_fn = control_message_mc.auto_count_fn();
    auto control_message               = std::make_shared<ControlMessage>();
    auto cm_meta = MessageMeta::create_from_cpp(std::move(create_cudf_table_with_metadata(20, 3)));
    control_message->payload(cm_meta);
    EXPECT_EQ((*control_message_auto_count_fn)(control_message), 20);

    // Test invalid message type
    EXPECT_THROW(MonitorController<int>("invalid message type"), std::runtime_error);
}
