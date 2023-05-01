/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define DOCA_ALLOW_EXPERIMENTAL_API

#include <doca_error.h>

#include <stdexcept>

namespace morpheus {

struct doca_error : public std::runtime_error {
  doca_error(std::string const& message) : std::runtime_error(message) {}
};

struct rte_error : public std::runtime_error {
  rte_error(std::string const& message) : std::runtime_error(message) {}
};

namespace detail {

inline void throw_doca_error(doca_error_t error, const char* file, unsigned int line)
{
  throw morpheus::doca_error(std::string{"DOCA error encountered at: " + std::string{file} + ":" +
                                     std::to_string(line) + ": " + std::to_string(error) + " " +
                                     std::string(doca_get_error_string(error))});
}

inline void throw_rte_error(int error, const char* file, unsigned int line)
{
  throw morpheus::rte_error(std::string{"RTE error encountered at: " + std::string{file} + ":" +
                                     std::to_string(line) + ": " + std::to_string(error)});
}

}

}

#define DOCA_TRY(call)                                                \
  do {                                                                \
    doca_error_t const status = (call);                               \
    if (DOCA_SUCCESS != status) {                                     \
      morpheus::detail::throw_doca_error(status, __FILE__, __LINE__); \
    }                                                                 \
  } while (0);

#define RTE_TRY(call)                                                 \
  do {                                                                \
    int const status = (call);                                        \
    if (status < 0) {                                                 \
      morpheus::detail::throw_rte_error(status, __FILE__, __LINE__);  \
    }                                                                 \
  } while (0);
