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
#include "morpheus/objects/file_types.hpp"

#include <boost/fiber/context.hpp>
#include <mrc/segment/builder.hpp>
#include <mrc/segment/object.hpp>
#include <pymrc/node.hpp>
#include <rxcpp/rx.hpp>

#include <fstream>
#include <functional>  // for function
#include <memory>
#include <string>
#include <thread>

namespace morpheus {
/****** Component public implementations *******************/
/****** WriteToFileStage********************************/

/**
 * @addtogroup stages
 * @{
 * @file
 */

/**
 * @brief Write all messages to a file. Messages are written to a file by this class.
 * This class does not maintain an open file or buffer messages.
 */
class MORPHEUS_EXPORT WriteToFileStage
  : public mrc::pymrc::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MessageMeta>>
{
  public:
    using base_t = mrc::pymrc::PythonNode<std::shared_ptr<MessageMeta>, std::shared_ptr<MessageMeta>>;
    using typename base_t::sink_type_t;
    using typename base_t::source_type_t;
    using typename base_t::subscribe_fn_t;

    /**
     * @brief Construct a new Write To File Stage object
     *
     * @param filename : Reference to the name of the file to which the messages will be written
     * @param mode : Reference to the mode for opening a file
     * @param file_type : FileTypes
     * @param include_index_col : Write out the index as a column, by default true
     * @param flush : When `true` flush the output buffer to disk on each message.
     */
    WriteToFileStage(const std::string& filename,
                     std::ios::openmode mode = std::ios::out,
                     FileTypes file_type     = FileTypes::Auto,
                     bool include_index_col  = true,
                     bool flush              = false);

  private:
    /**
     * @brief Close the file
     */
    void close();

    /**
     * @brief Write messages (rows in a DataFrame) to a JSON format
     *
     * @param msg
     */
    void write_json(sink_type_t& msg);

    /**
     * @brief Write messages (rows in a DataFrame) to a CSV format
     *
     * @param msg
     */
    void write_csv(sink_type_t& msg);

    /**
     * @brief Write messages (rows in a DataFrame) to a Parquet format
     *
     * @param msg
     */
    void write_parquet(sink_type_t& msg);

    subscribe_fn_t build_operator();

    bool m_is_first{};
    bool m_include_index_col;
    bool m_flush;
    std::ofstream m_fstream;
    std::function<void(sink_type_t&)> m_write_func;
};

/****** WriteToFileStageInterfaceProxy******************/
/**
 * @brief Interface proxy, used to insulate python bindings.
 */
struct MORPHEUS_EXPORT WriteToFileStageInterfaceProxy
{
    /**
     * @brief Create and initialize a WriteToFileStage, and return the result
     *
     * @param builder : Pipeline context object reference
     * @param name : Name of a stage reference
     * @param filename : Reference to the name of the file to which the messages will be written
     * @param mode : Reference to the mode for opening a file
     * @param file_type : FileTypes
     * @param include_index_col : Write out the index as a column, by default true
     * @param flush : When `true` flush the output buffer to disk on each message.
     * @return std::shared_ptr<mrc::segment::Object<WriteToFileStage>>
     */
    static std::shared_ptr<mrc::segment::Object<WriteToFileStage>> init(mrc::segment::Builder& builder,
                                                                        const std::string& name,
                                                                        const std::string& filename,
                                                                        const std::string& mode = "w",
                                                                        FileTypes file_type     = FileTypes::Auto,
                                                                        bool include_index_col  = true,
                                                                        bool flush              = false);
};
/** @} */  // end of group
}  // namespace morpheus
