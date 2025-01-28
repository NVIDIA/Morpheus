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

#pragma once

#include "morpheus/utilities/string_util.hpp"

#include <doca_error.h>

#include <stdexcept>

namespace morpheus {

struct DocaError : public std::runtime_error
{
    DocaError(std::string const& message) : std::runtime_error(message) {}
};

struct RteError : public std::runtime_error
{
    RteError(std::string const& message) : std::runtime_error(message) {}
};

namespace detail {

inline void throw_doca_error(doca_error_t error, const char* file, unsigned int line)
{
    throw morpheus::DocaError(MORPHEUS_CONCAT_STR("DOCA error encountered at: " << file << ":" << line << ": " << error
                                                                                << " " << doca_error_get_descr(error)));
}

inline void throw_rte_error(int error, const char* file, unsigned int line)
{
    throw morpheus::RteError(MORPHEUS_CONCAT_STR("RTE error encountered at: " << file << ":" << line << ": " << error));
}

}  // namespace detail

}  // namespace morpheus

#define DOCA_TRY(call)                                                        \
    do                                                                        \
    {                                                                         \
        doca_error_t ret_doca = (call);                                       \
        if (DOCA_SUCCESS != ret_doca)                                         \
        {                                                                     \
            morpheus::detail::throw_doca_error(ret_doca, __FILE__, __LINE__); \
        }                                                                     \
    } while (0);

#define RTE_TRY(call)                                                       \
    do                                                                      \
    {                                                                       \
        int ret_rte = (call);                                               \
        if (ret_rte < 0)                                                    \
        {                                                                   \
            morpheus::detail::throw_rte_error(ret_rte, __FILE__, __LINE__); \
        }                                                                   \
    } while (0);
