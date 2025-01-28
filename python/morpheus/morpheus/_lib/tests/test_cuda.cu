/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./test_utils/common.hpp"

#include <matx.h>

TEST_CLASS(Cuda);

TEST_F(TestCuda, LargeShape)
{
    // Test for issue #1004 Tensor shape with large dimensions, each dimension is < 2^31, but the total number of
    // elements is > 2^31 as is the number of bytes.
    const std::int32_t rows = 134217728;
    const std::int32_t cols = 4;
    auto tensor             = matx::make_tensor<float>({rows, cols});
    EXPECT_EQ(tensor.Size(0), rows);
    EXPECT_EQ(tensor.Size(1), cols);
}
