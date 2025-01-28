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

#include "morpheus/io/serializers.hpp"

#include "morpheus/objects/table_info_data.hpp"
#include "morpheus/utilities/cudf_util.hpp"

#include <cudf/io/csv.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>  // for column_name_info, sink_info, table_metadata
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <cstddef>  // for size_t
#include <fstream>
#include <numeric>
#include <sstream>  // IWYU pragma: keep
#include <vector>
// IWYU pragma: no_include <unordered_map>

namespace morpheus {
namespace py = pybind11;
using namespace py::literals;
using namespace std::string_literals;

class OStreamSink : public cudf::io::data_sink
{
  public:
    OStreamSink(std::ostream& stream) : m_stream(stream) {}

    /**
     * @brief Append the buffer content to the sink
     *
     * @param[in] data Pointer to the buffer to be written into the sink object
     * @param[in] size Number of bytes to write
     *
     * @return void
     */
    void host_write(void const* data, size_t size) override
    {
        m_stream.write(static_cast<char const*>(data), size);
        m_bytest_written += size;
    }

    /**
     * @brief Flush the data written into the sink
     */
    void flush() override
    {
        m_stream.flush();
    }

    /**
     * @brief Returns the total number of bytes written into this sink
     *
     * @return size_t Total number of bytes written into this sink
     */
    size_t bytes_written() override
    {
        return m_bytest_written;
    }

  private:
    std::ostream& m_stream;
    size_t m_bytest_written{0};
};

void table_to_csv(
    const TableInfoData& tbl, std::ostream& out_stream, bool include_header, bool include_index_col, bool flush)
{
    auto column_names         = tbl.column_names;
    cudf::size_type start_col = 1;
    if (include_index_col)
    {
        start_col = 0;
        column_names.insert(column_names.begin(), ""s);  // insert the id column
    }

    std::vector<cudf::size_type> col_idexes(column_names.size());
    std::iota(col_idexes.begin(), col_idexes.end(), start_col);
    auto tbl_view = tbl.table_view.select(col_idexes);

    OStreamSink sink(out_stream);
    auto destination     = cudf::io::sink_info(&sink);
    auto options_builder = cudf::io::csv_writer_options_builder(destination, tbl_view)
                               .include_header(include_header)
                               .true_value("True"s)
                               .false_value("False"s);

    if (include_header)
    {
        options_builder = options_builder.names(column_names);
    }

    cudf::io::write_csv(options_builder.build());

    if (flush)
    {
        sink.flush();
    }
}

void df_to_csv(const TableInfo& tbl, std::ostream& out_stream, bool include_header, bool include_index_col, bool flush)
{
    table_to_csv(TableInfoData{tbl.get_view(), tbl.get_index_names(), tbl.get_column_names()},
                 out_stream,
                 include_header,
                 include_index_col,
                 flush);
}

std::string df_to_csv(const TableInfo& tbl, bool include_header, bool include_index_col)
{
    // Create an ostringstream and use that with the overload accepting an ostream
    std::ostringstream out_stream;

    df_to_csv(tbl, out_stream, include_header, include_index_col);

    return out_stream.str();
}

void table_to_json(const TableInfoData& tbl, std::ostream& out_stream, bool include_index_col, bool flush)
{
    if (!include_index_col)
    {
        LOG(WARNING) << "Ignoring include_index_col=false as this isn't supported by cuDF";
    }

    auto column_names = tbl.column_names;
    std::vector<cudf::size_type> col_idexes(column_names.size());
    std::iota(col_idexes.begin(), col_idexes.end(), 1);
    auto tbl_view = tbl.table_view.select(col_idexes);

    cudf::io::table_metadata tbl_meta{
        std::vector<cudf::io::column_name_info>{column_names.cbegin(), column_names.cend()}};

    OStreamSink sink(out_stream);
    auto destination     = cudf::io::sink_info(&sink);
    auto options_builder = cudf::io::json_writer_options_builder(destination, tbl_view)
                               .metadata(tbl_meta)
                               .lines(true)
                               .include_nulls(true)
                               .na_rep("null");

    cudf::io::write_json(options_builder.build());

    if (flush)
    {
        sink.flush();
    }
}

void df_to_json(const TableInfo& tbl, std::ostream& out_stream, bool include_index_col, bool flush)
{
    table_to_json(TableInfoData{tbl.get_view(), tbl.get_index_names(), tbl.get_column_names()},
                  out_stream,
                  include_index_col,
                  flush);
}

std::string df_to_json(const TableInfo& tbl, bool include_index_col)
{
    // Create an ostringstream and use that with the overload accepting an ostream
    std::ostringstream out_stream;

    df_to_json(tbl, out_stream, include_index_col);

    return out_stream.str();
}

void table_to_parquet(
    const TableInfoData& tbl, std::ostream& out_stream, bool include_header, bool include_index_col, bool flush)
{
    auto column_names         = tbl.column_names;
    cudf::size_type start_col = 1;
    if (include_index_col)
    {
        start_col = 0;
        column_names.insert(column_names.begin(), ""s);  // insert the id column
    }

    std::vector<cudf::size_type> col_idexes(column_names.size());
    std::iota(col_idexes.begin(), col_idexes.end(), start_col);
    auto tbl_view = tbl.table_view.select(col_idexes);

    OStreamSink sink(out_stream);
    auto destination     = cudf::io::sink_info(&sink);
    auto options_builder = cudf::io::parquet_writer_options_builder(destination, tbl_view);

    cudf::io::write_parquet(options_builder.build());

    if (flush)
    {
        sink.flush();
    }
}

void df_to_parquet(
    const TableInfo& tbl, std::ostream& out_stream, bool include_header, bool include_index_col, bool flush)
{
    table_to_parquet(TableInfoData{tbl.get_view(), tbl.get_index_names(), tbl.get_column_names()},
                     out_stream,
                     include_header,
                     include_index_col,
                     flush);
}

std::string df_to_parquet(const TableInfo& tbl, bool include_header, bool include_index_col)
{
    // Create an ostringstream and use that with the overload accepting an ostream
    std::ostringstream out_stream;

    df_to_parquet(tbl, out_stream, include_header, include_index_col);

    return out_stream.str();
}

template <typename T>
T get_with_default(const py::dict& d, const std::string& key, T default_value)
{
    if (d.contains(key))
    {
        return d[key.c_str()].cast<T>();
    }

    return default_value;
}

void SerializersProxy::write_df_to_file(pybind11::object df,
                                        std::string filename,
                                        FileTypes file_type,
                                        const py::kwargs& kwargs)
{
    if (file_type == FileTypes::Auto)
    {
        file_type = determine_file_type(filename);  // throws if it is unable to determine the type
    }

    std::ofstream out_file;
    out_file.open(filename);

    auto tbl = CudfHelper::CudfHelper::table_info_data_from_table(df);

    switch (file_type)
    {
    case FileTypes::JSON: {
        table_to_json(tbl,
                      out_file,
                      get_with_default(kwargs, "include_index_col", true),
                      get_with_default(kwargs, "flush", false));
        break;
    }
    case FileTypes::CSV: {
        table_to_csv(tbl,
                     out_file,
                     get_with_default(kwargs, "include_header", true),
                     get_with_default(kwargs, "include_index_col", true),
                     get_with_default(kwargs, "flush", false));
        break;
    }
    case FileTypes::Auto:
    default:
        throw std::logic_error(MORPHEUS_CONCAT_STR("Unsupported filetype: " << file_type));
    }
}
}  // namespace morpheus
