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

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace morpheus {
/**
 * @addtogroup utility_error
 * @{
 * @file
 */

/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * MORPHEUS_EXPECTS macro.
 */
struct LogicError : public std::logic_error
{
    LogicError(char const* const message) : std::logic_error(message) {}

    LogicError(std::string const& message) : std::logic_error(message) {}
};
/**
 * @brief Exception thrown when a CUDA error is encountered.
 */
struct CudaError : public std::runtime_error
{
    CudaError(std::string const& message) : std::runtime_error(message) {}
};
/** @} */

}  // namespace morpheus

#define STRINGIFY_DETAIL(x) #x
#define MORPHEUS_STRINGIFY(x) STRINGIFY_DETAIL(x)

/**
 * @addtogroup utility_error
 * @{
 */

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when
 * a condition is violated.
 *
 * Example usage:
 *
 * @code
 * MORPHEUS_EXPECTS(lhs->dtype == rhs->dtype, "Column type mismatch");
 * @endcode
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] reason String literal description of the reason that cond is
 * expected to be true
 * @throw morpheus::LogicError if the condition evaluates to false.
 */
#define MORPHEUS_EXPECTS(cond, reason) \
    (!!(cond))                         \
        ? static_cast<void>(0)         \
        : throw morpheus::LogicError("Morpheus failure at: " __FILE__ ":" MORPHEUS_STRINGIFY(__LINE__) ": " reason)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * In host code, throws a `morpheus::LogicError`.
 *
 *
 * Example usage:
 * ```
 * MORPHEUS_FAIL("Non-arithmetic operation is not supported");
 * ```
 *
 * @param[in] reason String literal description of the reason
 */
#define MORPHEUS_FAIL(reason) \
    throw morpheus::LogicError("Morpheus failure at: " __FILE__ ":" MORPHEUS_STRINGIFY(__LINE__) ": " reason)

namespace morpheus::detail {

inline void throw_cuda_error(cudaError_t error, const char* file, unsigned int line)
{
    throw morpheus::CudaError(std::string{"CUDA error encountered at: " + std::string{file} + ":" +
                                          std::to_string(line) + ": " + std::to_string(error) + " " +
                                          cudaGetErrorName(error) + " " + cudaGetErrorString(error)});
}
}  // namespace morpheus::detail

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 */
#define CUDA_TRY(call)                                                      \
    do                                                                      \
    {                                                                       \
        cudaError_t const status = (call);                                  \
        if (cudaSuccess != status)                                          \
        {                                                                   \
            cudaGetLastError();                                             \
            morpheus::detail::throw_cuda_error(status, __FILE__, __LINE__); \
        }                                                                   \
    } while (0);

/**
 * @brief Debug macro to check for CUDA errors
 *
 * In a non-release build, this macro will synchronize the specified stream
 * before error checking. In both release and non-release builds, this macro
 * checks for any pending CUDA errors from previous calls. If an error is
 * reported, an exception is thrown detailing the CUDA error that occurred.
 *
 * The intent of this macro is to provide a mechanism for synchronous and
 * deterministic execution for debugging asynchronous CUDA execution. It should
 * be used after any asynchronous CUDA call, e.g., cudaMemcpyAsync, or an
 * asynchronous kernel launch.
 */
#define CHECK_CUDA(stream)                       \
    do                                           \
    {                                            \
        CUDA_TRY(cudaStreamSynchronize(stream)); \
        CUDA_TRY(cudaPeekAtLastError());         \
    } while (0);
/** @} */
