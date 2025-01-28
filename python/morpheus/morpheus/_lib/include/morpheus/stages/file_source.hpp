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

#pragma once

#include "morpheus/export.h"
#include "morpheus/messages/meta.hpp"

#include <boost/fiber/context.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <pybind11/pytypes.h>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>  // for apply, make_subscriber, observable_member, is_on_error<>::not_void, is_on_next_of<>::not_void, trace_activity

#include <filesystem>  // for path
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace morpheus {
/****** Component public implementations *******************/
/****** FileSourceStage*************************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

/**
 * @brief Load messages from a file. Source stage is used to load messages from a file and
 * dumping the contents into the pipeline immediately. Useful for testing performance and accuracy of a pipeline.
 */
class MORPHEUS_EXPORT FileSourceStage : public mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = mrc::pymrc::PythonSource<std::shared_ptr<MessageMeta>>;
    using typename base_t::source_type_t;
    using typename base_t::subscriber_fn_t;

    /**
     * @brief Construct a new File Source Stage object
     *
     * @param filename : Name of the file from which the messages will be read
     * @param repeat : Repeats the input dataset multiple times. Useful to extend small datasets for debugging
     * @param json_lines: Whether to force json or jsonlines parsing
     */
    FileSourceStage(std::string filename,
                    int repeat                                   = 1,
                    bool filter_null                             = true,
                    std::vector<std::string> filter_null_columns = {},
                    std::optional<bool> json_lines               = std::nullopt);

  private:
    subscriber_fn_t build();

    std::string m_filename;
    int m_repeat{1};
    bool m_filter_null{true};
    std::vector<std::string> m_filter_null_columns;
    std::optional<bool> m_json_lines;
};

/****** FileSourceStageInterfaceProxy***********************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT FileSourceStageInterfaceProxy
{
    /**
     * @brief Create and initialize a FileSourceStage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param filename : Name of the file from which the messages will be read.
     * @param repeat : Repeats the input dataset multiple times. Useful to extend small datasets for debugging.
     * @param parser_kwargs : Optional arguments to pass to the file parser.
     * @return std::shared_ptr<mrc::segment::Object<FileSourceStage>>
     */
    static std::shared_ptr<mrc::segment::Object<FileSourceStage>> init(
        mrc::segment::Builder& builder,
        const std::string& name,
        std::string filename,
        int repeat                                   = 1,
        bool filter_null                             = true,
        std::vector<std::string> filter_null_columns = {},
        pybind11::dict parser_kwargs                 = pybind11::dict());
    static std::shared_ptr<mrc::segment::Object<FileSourceStage>> init(
        mrc::segment::Builder& builder,
        const std::string& name,
        std::filesystem::path filename,
        int repeat                                   = 1,
        bool filter_null                             = true,
        std::vector<std::string> filter_null_columns = {},
        pybind11::dict parser_kwargs                 = pybind11::dict());
};
/** @} */  // end of group
}  // namespace morpheus
