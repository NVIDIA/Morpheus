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

#include "../test_utils/common.hpp"
#include "../test_utils/tensor_utils.hpp"  // for assert_eq_device_to_host
#include "test_messages.hpp"               // for TestMessages

#include "morpheus/io/deserializers.hpp"    // for load_table_from_file
#include "morpheus/messages/meta.hpp"       // for MessageMeta
#include "morpheus/objects/dtype.hpp"       // for DType
#include "morpheus/objects/table_info.hpp"  // for TableInfo
#include "morpheus/objects/tensor.hpp"      // for Tensor
#include "morpheus/types.hpp"               // for RangeType

#include <gtest/gtest.h>   // for TestInfo, TEST_F
#include <pybind11/gil.h>  // for gil_scoped_release
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rmm/cuda_stream_view.hpp>  // for cuda_stream_per_thread
#include <rmm/device_buffer.hpp>     // for device_buffer

#include <cstdint>     // for int64_t
#include <filesystem>  // for operator/, path
#include <memory>      // for allocator, __shared_ptr_access, shared_ptr, make_shared
#include <utility>     // for move
#include <vector>      // for vector

using namespace morpheus;
using namespace morpheus::test;
using namespace pybind11::literals;

using TestMessageMeta = morpheus::test::TestMessages;  // NOLINT(readability-identifier-naming)

TEST_F(TestMessageMeta, SetdataWithColumnName)
{
    pybind11::gil_scoped_release no_gil;
    auto test_data_dir               = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "csv_sample.csv";

    auto meta = MessageMeta::create_from_cpp(load_table_from_file(input_file));

    std::vector<int64_t> packed_data_host{9, 8, 7, 6, 5, 4, 3, 2, 1};
    int mess_count   = packed_data_host.size();
    int cols_size    = 1;
    auto packed_data = std::make_shared<rmm::device_buffer>(
        packed_data_host.data(), mess_count * cols_size * sizeof(int64_t), rmm::cuda_stream_per_thread);

    auto tensor = Tensor::create(packed_data, DType::create<int64_t>(), {mess_count, cols_size}, {}, 0);
    meta->set_data("int", tensor);

    assert_eq_device_to_host(meta->get_info().get_column(0), packed_data_host);
}

TEST_F(TestMessageMeta, CopyRangeAndSlicing)
{
    pybind11::gil_scoped_release no_gil;
    auto test_data_dir               = test::get_morpheus_root() / "tests/tests_data";
    std::filesystem::path input_file = test_data_dir / "csv_sample.csv";

    auto meta = MessageMeta::create_from_cpp(load_table_from_file(input_file));

    std::vector<RangeType> ranges                  = {{0, 1}, {3, 6}};
    auto copy_range_meta                           = meta->copy_ranges(ranges);
    std::vector<int64_t> copy_range_expected_int   = {1, 4, 5, 6};
    std::vector<double> copy_range_expected_double = {1.1, 4.4, 5.5, 6.6};
    assert_eq_device_to_host(copy_range_meta->get_info().get_column(0), copy_range_expected_int);
    assert_eq_device_to_host(copy_range_meta->get_info().get_column(1), copy_range_expected_double);

    auto sliced_meta                           = meta->get_slice(2, 4);
    std::vector<int64_t> sliced_expected_int   = {3, 4};
    std::vector<double> sliced_expected_double = {3.3, 4.4};
    assert_eq_device_to_host(sliced_meta->get_info().get_column(0), sliced_expected_int);
    assert_eq_device_to_host(sliced_meta->get_info().get_column(1), sliced_expected_double);
}

TEST_F(TestMessageMeta, Issue1934)
{
    // Reproduce issue 1934 (https://github.com/nv-morpheus/Morpheus/issues/1934)
    // The bug causes a segfault when calling `get_data_frame` on a message meta object
    namespace py = pybind11;
    py::gil_scoped_acquire gil;

    auto cudf_mod = py::module_::import("cudf");
    auto a_col    = py::list();
    auto v1       = py::list();
    v1.append(py::str("a"));
    a_col.attr("append")(std::move(v1));
    a_col.attr("append")(py::none());

    auto df = cudf_mod.attr("DataFrame")(py::dict("a"_a = std::move(a_col)));
    df.attr("drop")(0, "inplace"_a = true);

    auto msg = MessageMetaInterfaceProxy::init_python(std::move(df));

    auto df_copy = MessageMetaInterfaceProxy::get_data_frame(*msg);
}
