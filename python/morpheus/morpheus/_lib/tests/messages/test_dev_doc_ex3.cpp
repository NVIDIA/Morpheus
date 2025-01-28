/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "morpheus/messages/control.hpp"    // for ControlMessage
#include "morpheus/messages/meta.hpp"       // for MessageMeta
#include "morpheus/objects/table_info.hpp"  // for MutableTableInfo

#include <gtest/gtest.h>
#include <pybind11/gil.h>       // for gil_scoped_release, gil_scoped_acquire
#include <pybind11/pybind11.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>   // for object, object_api, literals

#include <memory>   // for shared_ptr
#include <utility>  // for move

using namespace morpheus;
using namespace morpheus::test;

class TestDevDocEx3 : public morpheus::test::TestWithPythonInterpreter
{};

TEST_F(TestDevDocEx3, TestPyObjFromMultiMesg)
{
    // Test to verify that the lambda in docs/source/developer_guide/guides/3_simple_cpp_stage.md
    // compiles and runs correctly.
    // If you have to make a change here to ensure the test passes, please update the documentation
    using namespace pybind11::literals;
    pybind11::gil_scoped_release no_gil;

    auto doc_fn = [](std::shared_ptr<ControlMessage> msg) {
        auto mutable_info = msg->payload()->get_mutable_info();

        std::shared_ptr<MessageMeta> new_meta;
        {
            pybind11::gil_scoped_acquire gil;
            auto df = mutable_info.checkout_obj();

            // Maka a copy of the original DataFrame
            auto copied_df = df->attr("copy")("deep"_a = true);

            // Now that we are done with `df` return it to the owner
            mutable_info.return_obj(std::move(df));

            // do something with copied_df
            new_meta = MessageMeta::create_from_python(std::move(copied_df));
        }  // GIL is released

        return new_meta;
    };

    auto msg_meta = create_mock_msg_meta({"col1", "col2", "col3"}, {"int32", "float32", "string"}, 5);
    auto msg      = std::make_shared<ControlMessage>();
    msg->payload(msg_meta);

    auto result = doc_fn(msg);

    auto orig_mutable_info   = msg_meta->get_mutable_info();
    auto result_mutable_info = result->get_mutable_info();
    {
        pybind11::gil_scoped_acquire gil;

        auto orig_df   = orig_mutable_info.checkout_obj();
        auto result_df = result_mutable_info.checkout_obj();

        // orig_df.eq(result_df).all().all()
        auto is_true = orig_df->attr("eq")(*result_df).attr("all")().attr("all")();
        EXPECT_TRUE(is_true.cast<bool>());

        orig_mutable_info.return_obj(std::move(orig_df));
        result_mutable_info.return_obj(std::move(result_df));
    }
}
