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

#include "../test_utils/common.hpp"  // IWYU pragma: associated

#include "morpheus/objects/dtype.hpp"  // for DType

#include <cudf/types.hpp>
#include <gtest/gtest.h>

#include <cstdint>  // for int32_t
#include <set>      // for set
#include <stdexcept>

using namespace morpheus;
using namespace morpheus::test;

TEST_CLASS(DType);

TEST_F(TestDType, FromNumpyValidStr)
{
    DType dtype = DType::from_numpy("|i1");
    ASSERT_EQ(dtype.type_id(), TypeId::INT8);
    ASSERT_EQ(dtype.item_size(), 1);
    ASSERT_EQ(dtype.type_str(), "|i1");

    dtype = DType::from_numpy("<i2");
    ASSERT_EQ(dtype.type_id(), TypeId::INT16);
    ASSERT_EQ(dtype.item_size(), 2);
    ASSERT_EQ(dtype.type_str(), "<i2");

    dtype = DType::from_numpy("<i4");
    ASSERT_EQ(dtype.type_id(), TypeId::INT32);
    ASSERT_EQ(dtype.item_size(), 4);
    ASSERT_EQ(dtype.type_str(), "<i4");

    dtype = DType::from_numpy("<i8");
    ASSERT_EQ(dtype.type_id(), TypeId::INT64);
    ASSERT_EQ(dtype.item_size(), 8);
    ASSERT_EQ(dtype.type_str(), "<i8");

    dtype = DType::from_numpy("|u1");
    ASSERT_EQ(dtype.type_id(), TypeId::UINT8);
    ASSERT_EQ(dtype.item_size(), 1);
    ASSERT_EQ(dtype.type_str(), "|u1");

    dtype = DType::from_numpy("<u2");
    ASSERT_EQ(dtype.type_id(), TypeId::UINT16);
    ASSERT_EQ(dtype.item_size(), 2);
    ASSERT_EQ(dtype.type_str(), "<u2");

    dtype = DType::from_numpy("<u4");
    ASSERT_EQ(dtype.type_id(), TypeId::UINT32);
    ASSERT_EQ(dtype.item_size(), 4);
    ASSERT_EQ(dtype.type_str(), "<u4");

    dtype = DType::from_numpy("<u8");
    ASSERT_EQ(dtype.type_id(), TypeId::UINT64);
    ASSERT_EQ(dtype.item_size(), 8);
    ASSERT_EQ(dtype.type_str(), "<u8");

    dtype = DType::from_numpy("|b1");
    ASSERT_EQ(dtype.type_id(), TypeId::BOOL8);
    ASSERT_EQ(dtype.item_size(), 1);
    ASSERT_EQ(dtype.type_str(), "|b1");

    dtype = DType::from_numpy("<f4");
    ASSERT_EQ(dtype.type_id(), TypeId::FLOAT32);
    ASSERT_EQ(dtype.item_size(), 4);
    ASSERT_EQ(dtype.type_str(), "<f4");

    dtype = DType::from_numpy("<f8");
    ASSERT_EQ(dtype.type_id(), TypeId::FLOAT64);
    ASSERT_EQ(dtype.item_size(), 8);
    ASSERT_EQ(dtype.type_str(), "<f8");
}

TEST_F(TestDType, FromNumpyInvalidStr)
{
    // invalid byte order char
    EXPECT_THROW(DType::from_numpy("{i2"), std::invalid_argument);

    // invalid type char
    EXPECT_THROW(DType::from_numpy("<x2"), std::invalid_argument);

    // invalid byte size
    EXPECT_THROW(DType::from_numpy("<i9"), std::invalid_argument);

    // all invalid
    EXPECT_THROW(DType::from_numpy("{x9"), std::invalid_argument);
    EXPECT_THROW(DType::from_numpy("zzzzz"), std::invalid_argument);
    EXPECT_THROW(DType::from_numpy("123456"), std::invalid_argument);
    EXPECT_THROW(DType::from_numpy("*&()#%"), std::invalid_argument);

    // empty string
    EXPECT_THROW(DType::from_numpy(""), std::invalid_argument);
}

TEST_F(TestDType, FromTritonValidStr)
{
    DType dtype = DType::from_triton("INT8");
    ASSERT_EQ(dtype.type_id(), TypeId::INT8);
    ASSERT_EQ(dtype.triton_str(), "INT8");
    ASSERT_EQ(dtype.item_size(), 1);
    ASSERT_EQ(dtype.type_str(), "|i1");

    dtype = DType::from_triton("INT16");
    ASSERT_EQ(dtype.type_id(), TypeId::INT16);
    ASSERT_EQ(dtype.triton_str(), "INT16");
    ASSERT_EQ(dtype.item_size(), 2);
    ASSERT_EQ(dtype.type_str(), "<i2");

    dtype = DType::from_triton("INT32");
    ASSERT_EQ(dtype.type_id(), TypeId::INT32);
    ASSERT_EQ(dtype.triton_str(), "INT32");
    ASSERT_EQ(dtype.item_size(), 4);
    ASSERT_EQ(dtype.type_str(), "<i4");

    dtype = DType::from_triton("INT64");
    ASSERT_EQ(dtype.type_id(), TypeId::INT64);
    ASSERT_EQ(dtype.triton_str(), "INT64");
    ASSERT_EQ(dtype.item_size(), 8);
    ASSERT_EQ(dtype.type_str(), "<i8");

    dtype = DType::from_triton("UINT8");
    ASSERT_EQ(dtype.type_id(), TypeId::UINT8);
    ASSERT_EQ(dtype.triton_str(), "UINT8");
    ASSERT_EQ(dtype.item_size(), 1);
    ASSERT_EQ(dtype.type_str(), "|u1");

    dtype = DType::from_triton("UINT16");
    ASSERT_EQ(dtype.type_id(), TypeId::UINT16);
    ASSERT_EQ(dtype.triton_str(), "UINT16");
    ASSERT_EQ(dtype.item_size(), 2);
    ASSERT_EQ(dtype.type_str(), "<u2");

    dtype = DType::from_triton("UINT32");
    ASSERT_EQ(dtype.type_id(), TypeId::UINT32);
    ASSERT_EQ(dtype.triton_str(), "UINT32");
    ASSERT_EQ(dtype.item_size(), 4);
    ASSERT_EQ(dtype.type_str(), "<u4");

    dtype = DType::from_triton("UINT64");
    ASSERT_EQ(dtype.type_id(), TypeId::UINT64);
    ASSERT_EQ(dtype.triton_str(), "UINT64");
    ASSERT_EQ(dtype.item_size(), 8);
    ASSERT_EQ(dtype.type_str(), "<u8");

    dtype = DType::from_triton("BOOL");
    ASSERT_EQ(dtype.type_id(), TypeId::BOOL8);
    ASSERT_EQ(dtype.triton_str(), "BOOL");
    ASSERT_EQ(dtype.item_size(), 1);
    ASSERT_EQ(dtype.type_str(), "|b1");

    dtype = DType::from_triton("FP32");
    ASSERT_EQ(dtype.type_id(), TypeId::FLOAT32);
    ASSERT_EQ(dtype.triton_str(), "FP32");
    ASSERT_EQ(dtype.item_size(), 4);
    ASSERT_EQ(dtype.type_str(), "<f4");

    dtype = DType::from_triton("FP64");
    ASSERT_EQ(dtype.type_id(), TypeId::FLOAT64);
    ASSERT_EQ(dtype.triton_str(), "FP64");
    ASSERT_EQ(dtype.item_size(), 8);
    ASSERT_EQ(dtype.type_str(), "<f8");
}

TEST_F(TestDType, FromTritonInvalidStr)
{
    EXPECT_THROW(DType::from_triton("BOOL8"), std::invalid_argument);
    EXPECT_THROW(DType::from_triton("FLOAT32"), std::invalid_argument);
    EXPECT_THROW(DType::from_triton("FLOAT64"), std::invalid_argument);
    EXPECT_THROW(DType::from_triton("uint8"), std::invalid_argument);
    EXPECT_THROW(DType::from_triton("zzzzz"), std::invalid_argument);
    EXPECT_THROW(DType::from_triton("123456"), std::invalid_argument);
    EXPECT_THROW(DType::from_triton("*&()#%"), std::invalid_argument);
    EXPECT_THROW(DType::from_triton(""), std::invalid_argument);
}

TEST_F(TestDType, FromCudfSupported)
{
    DType dtype = DType::from_cudf(cudf::type_id::INT8);
    ASSERT_EQ(dtype.type_id(), TypeId::INT8);
    ASSERT_EQ(dtype.cudf_type_id(), cudf::type_id::INT8);
    ASSERT_EQ(dtype.item_size(), 1);
    ASSERT_EQ(dtype.type_str(), "|i1");

    dtype = DType::from_cudf(cudf::type_id::INT16);
    ASSERT_EQ(dtype.type_id(), TypeId::INT16);
    ASSERT_EQ(dtype.cudf_type_id(), cudf::type_id::INT16);
    ASSERT_EQ(dtype.item_size(), 2);
    ASSERT_EQ(dtype.type_str(), "<i2");

    dtype = DType::from_cudf(cudf::type_id::INT32);
    ASSERT_EQ(dtype.type_id(), TypeId::INT32);
    ASSERT_EQ(dtype.cudf_type_id(), cudf::type_id::INT32);
    ASSERT_EQ(dtype.item_size(), 4);
    ASSERT_EQ(dtype.type_str(), "<i4");

    dtype = DType::from_cudf(cudf::type_id::INT64);
    ASSERT_EQ(dtype.type_id(), TypeId::INT64);
    ASSERT_EQ(dtype.cudf_type_id(), cudf::type_id::INT64);
    ASSERT_EQ(dtype.item_size(), 8);
    ASSERT_EQ(dtype.type_str(), "<i8");

    dtype = DType::from_cudf(cudf::type_id::UINT8);
    ASSERT_EQ(dtype.type_id(), TypeId::UINT8);
    ASSERT_EQ(dtype.cudf_type_id(), cudf::type_id::UINT8);
    ASSERT_EQ(dtype.item_size(), 1);
    ASSERT_EQ(dtype.type_str(), "|u1");

    dtype = DType::from_cudf(cudf::type_id::UINT16);
    ASSERT_EQ(dtype.type_id(), TypeId::UINT16);
    ASSERT_EQ(dtype.cudf_type_id(), cudf::type_id::UINT16);
    ASSERT_EQ(dtype.item_size(), 2);
    ASSERT_EQ(dtype.type_str(), "<u2");

    dtype = DType::from_cudf(cudf::type_id::UINT32);
    ASSERT_EQ(dtype.type_id(), TypeId::UINT32);
    ASSERT_EQ(dtype.cudf_type_id(), cudf::type_id::UINT32);
    ASSERT_EQ(dtype.item_size(), 4);
    ASSERT_EQ(dtype.type_str(), "<u4");

    dtype = DType::from_cudf(cudf::type_id::UINT64);
    ASSERT_EQ(dtype.type_id(), TypeId::UINT64);
    ASSERT_EQ(dtype.cudf_type_id(), cudf::type_id::UINT64);
    ASSERT_EQ(dtype.item_size(), 8);
    ASSERT_EQ(dtype.type_str(), "<u8");

    dtype = DType::from_cudf(cudf::type_id::BOOL8);
    ASSERT_EQ(dtype.type_id(), TypeId::BOOL8);
    ASSERT_EQ(dtype.cudf_type_id(), cudf::type_id::BOOL8);
    ASSERT_EQ(dtype.item_size(), 1);
    ASSERT_EQ(dtype.type_str(), "|b1");

    dtype = DType::from_cudf(cudf::type_id::FLOAT32);
    ASSERT_EQ(dtype.type_id(), TypeId::FLOAT32);
    ASSERT_EQ(dtype.cudf_type_id(), cudf::type_id::FLOAT32);
    ASSERT_EQ(dtype.item_size(), 4);
    ASSERT_EQ(dtype.type_str(), "<f4");

    dtype = DType::from_cudf(cudf::type_id::FLOAT64);
    ASSERT_EQ(dtype.type_id(), TypeId::FLOAT64);
    ASSERT_EQ(dtype.cudf_type_id(), cudf::type_id::FLOAT64);
    ASSERT_EQ(dtype.item_size(), 8);
    ASSERT_EQ(dtype.type_str(), "<f8");

    dtype = DType::from_cudf(cudf::type_id::STRING);
    ASSERT_EQ(dtype.type_id(), TypeId::STRING);
    ASSERT_EQ(dtype.cudf_type_id(), cudf::type_id::STRING);
    ASSERT_EQ(dtype.item_size(), 1);
    // numpy typestr for string not supported
    EXPECT_THROW(dtype.type_str(), std::runtime_error);
}

TEST_F(TestDType, FromCudfNotSupported)
{
    EXPECT_THROW(DType::from_cudf(cudf::type_id::TIMESTAMP_DAYS), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::TIMESTAMP_SECONDS), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::TIMESTAMP_MILLISECONDS), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::TIMESTAMP_MICROSECONDS), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::TIMESTAMP_NANOSECONDS), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::DURATION_DAYS), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::DURATION_SECONDS), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::DURATION_MILLISECONDS), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::DURATION_NANOSECONDS), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::DICTIONARY32), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::LIST), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::DECIMAL32), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::DECIMAL64), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::DECIMAL128), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::STRUCT), std::invalid_argument);
    EXPECT_THROW(DType::from_cudf(cudf::type_id::NUM_TYPE_IDS), std::invalid_argument);
}

TEST_F(TestDType, IsFullySupported)
{
    std::set<TypeId> unsupported_types = {TypeId::EMPTY, TypeId::STRING, TypeId::NUM_TYPE_IDS};
    for (auto type_id = static_cast<int32_t>(TypeId::EMPTY); type_id <= static_cast<int32_t>(TypeId::NUM_TYPE_IDS);
         ++type_id)
    {
        auto enum_type_id = static_cast<TypeId>(type_id);
        auto dtype        = DType(enum_type_id);

        ASSERT_EQ(dtype.is_fully_supported(), !unsupported_types.contains(enum_type_id));
    }
}
