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

#include "./test_utils/common.hpp"  // IWYU pragma: associated

#include "morpheus/objects/dtype.hpp"

#include <gtest/gtest.h>  // for EXPECT_EQ

#include <vector>
// work-around for known iwyu issue
// https://github.com/include-what-you-use/include-what-you-use/issues/908
// IWYU pragma: no_include <algorithm>

TEST_CLASS(TypeUtils);

TEST_F(TestTypeUtils, DTypeCopy)
{
    morpheus::DType d1(morpheus::TypeId::INT32);
    morpheus::DType d2(morpheus::TypeId::FLOAT32);

    std::vector<morpheus::DType> type_list;
    type_list.push_back(d1);
    type_list.push_back(d2);
    type_list.emplace_back(morpheus::TypeId::INT32);
    type_list.emplace_back(morpheus::TypeId::FLOAT32);

    EXPECT_EQ(type_list[0], d1);
    EXPECT_EQ(type_list[1], d2);
    EXPECT_EQ(type_list[2], d1);
    EXPECT_EQ(type_list[3], d2);

    morpheus::DType d3 = d1;
    morpheus::DType d4 = d2;

    EXPECT_EQ(d3, d1);
    EXPECT_EQ(d3.type_id(), d1.type_id());

    EXPECT_EQ(d4, d2);
    EXPECT_EQ(d4.type_id(), d2.type_id());

    morpheus::DType d5{d1};
    morpheus::DType d6{d2};
}
