# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools

import cudf
from cudf.core.column import ColumnBase
from cudf.core.dtypes import StructDtype

from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector
from pylibcudf.libcudf.io.types cimport column_name_info
from pylibcudf.libcudf.io.types cimport table_metadata
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.table.table_view cimport table_view
from pylibcudf.libcudf.types cimport size_type

##### THE FOLLOWING CODE IS COPIED FROM CUDF AND SHOULD BE REMOVED WHEN UPDATING TO cudf>=24.12 #####
# see https://github.com/rapidsai/cudf/pull/17193 for details

# isort: off

# imports needed for get_element, which is required by from_column_view_with_fix
cimport pylibcudf.libcudf.copying as cpp_copying
from pylibcudf.libcudf.column.column_view cimport column_view
from libcpp.memory cimport make_unique, unique_ptr
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf cimport Table as plc_Table, Scalar as plc_Scalar
import pylibcudf as plc

# imports needed for from_column_view_with_fix
import rmm
from libc.stdint cimport uintptr_t
from cudf.core.buffer import (
    # Buffer,
    ExposureTrackedBuffer,
    SpillableBuffer,
    # acquire_spill_lock,
    as_buffer,
    # cuda_array_interface_wrapper,
)
cimport pylibcudf.libcudf.types as libcudf_types
from cudf._lib.types cimport (
    dtype_from_column_view,
    # dtype_to_data_type,
    # dtype_to_pylibcudf_type,
)
from cudf._lib.null_mask import bitmask_allocation_size_bytes
from cudf._lib.column cimport Column
# isort: on

cdef get_element(column_view col_view, size_type index):

    cdef unique_ptr[scalar] c_output
    with nogil:
        c_output = move(
            cpp_copying.get_element(col_view, index)
        )

    plc_scalar = plc_Scalar.from_libcudf(move(c_output))
    return plc.interop.to_arrow(plc_scalar).as_py()


cdef Column from_column_view_with_fix(column_view cv, object owner):
    """
    Given a ``cudf::column_view``, constructs a ``cudf.Column`` from it,
    along with referencing an ``owner`` Python object that owns the memory
    lifetime. If ``owner`` is a ``cudf.Column``, we reach inside of it and
    make the owner of each newly created ``Buffer`` the respective
    ``Buffer`` from the ``owner`` ``cudf.Column``.
    If ``owner`` is ``None``, we allocate new memory for the resulting
    ``cudf.Column``.
    """
    column_owner = isinstance(owner, Column)
    mask_owner = owner
    if column_owner and isinstance(owner.dtype, cudf.CategoricalDtype):
        owner = owner.base_children[0]

    size = cv.size()
    offset = cv.offset()
    dtype = dtype_from_column_view(cv)
    dtype_itemsize = getattr(dtype, "itemsize", 1)

    data_ptr = <uintptr_t>(cv.head[void]())
    data = None
    base_size = size + offset
    data_owner = owner

    if column_owner:
        data_owner = owner.base_data
        mask_owner = mask_owner.base_mask
        base_size = owner.base_size
    base_nbytes = base_size * dtype_itemsize
    # special case for string column
    is_string_column = (cv.type().id() == libcudf_types.type_id.STRING)
    if is_string_column:
        if cv.num_children() == 0:
            base_nbytes = 0
        else:
            # get the size from offset child column (device to host copy)
            offsets_column_index = 0
            offset_child_column = cv.child(offsets_column_index)
            if offset_child_column.size() == 0:
                base_nbytes = 0
            else:
                chars_size = get_element(
                    offset_child_column, offset_child_column.size()-1)
                base_nbytes = chars_size

    if data_ptr:
        if data_owner is None:
            buffer_size = (
                base_nbytes
                if is_string_column
                else ((size + offset) * dtype_itemsize)
            )
            data = as_buffer(
                rmm.DeviceBuffer(ptr=data_ptr,
                                    size=buffer_size)
            )
        elif (
            column_owner and
            isinstance(data_owner, ExposureTrackedBuffer)
        ):
            data = as_buffer(
                data=data_ptr,
                size=base_nbytes,
                owner=data_owner,
                exposed=False,
            )
        elif (
            # This is an optimization of the most common case where
            # from_column_view creates a "view" that is identical to
            # the owner.
            column_owner and
            isinstance(data_owner, SpillableBuffer) and
            # We check that `data_owner` is spill locked (not spillable)
            # and that it points to the same memory as `data_ptr`.
            not data_owner.spillable and
            data_owner.memory_info() == (data_ptr, base_nbytes, "gpu")
        ):
            data = data_owner
        else:
            # At this point we don't know the relationship between data_ptr
            # and data_owner thus we mark both of them exposed.
            # TODO: try to discover their relationship and create a
            #       SpillableBufferSlice instead.
            data = as_buffer(
                data=data_ptr,
                size=base_nbytes,
                owner=data_owner,
                exposed=True,
            )
            if isinstance(data_owner, ExposureTrackedBuffer):
                # accessing the pointer marks it exposed permanently.
                data_owner.mark_exposed()
            elif isinstance(data_owner, SpillableBuffer):
                if data_owner.is_spilled:
                    raise ValueError(
                        f"{data_owner} is spilled, which invalidates "
                        f"the exposed data_ptr ({hex(data_ptr)})"
                    )
                # accessing the pointer marks it exposed permanently.
                data_owner.mark_exposed()
    else:
        data = as_buffer(
            rmm.DeviceBuffer(ptr=data_ptr, size=0)
        )

    mask = None
    mask_ptr = <uintptr_t>(cv.null_mask())
    if mask_ptr:
        if mask_owner is None:
            if column_owner:
                # if we reached here, it means `owner` is a `Column`
                # that does not have a null mask, but `cv` thinks it
                # should have a null mask. This can happen in the
                # following sequence of events:
                #
                # 1) `cv` is constructed as a view into a
                #    `cudf::column` that is nullable (i.e., it has
                #    a null mask), but contains no nulls.
                # 2) `owner`, a `Column`, is constructed from the
                #    same `cudf::column`. Because `cudf::column`
                #    is memory owning, `owner` takes ownership of
                #    the memory owned by the
                #    `cudf::column`. Because the column has a null
                #    count of 0, it may choose to discard the null
                #    mask.
                # 3) Now, `cv` points to a discarded null mask.
                #
                # TL;DR: we should not include a null mask in the
                # result:
                mask = None
            else:
                mask = as_buffer(
                    rmm.DeviceBuffer(
                        ptr=mask_ptr,
                        size=bitmask_allocation_size_bytes(base_size)
                    )
                )
        else:
            mask = as_buffer(
                data=mask_ptr,
                size=bitmask_allocation_size_bytes(base_size),
                owner=mask_owner,
                exposed=True
            )

    if cv.has_nulls():
        null_count = cv.null_count()
    else:
        null_count = 0

    children = []
    for child_index in range(cv.num_children()):
        child_owner = owner
        if column_owner:
            child_owner = owner.base_children[child_index]
        children.append(
            from_column_view_with_fix(
                cv.child(child_index),
                child_owner
            )
        )
    children = tuple(children)

    result = cudf.core.column.build_column(
        data=data,
        dtype=dtype,
        mask=mask,
        size=size,
        offset=offset,
        null_count=null_count,
        children=tuple(children)
    )

    return result

##### THE PREVIOUS CODE IS COPIED FROM CUDF AND SHOULD BE REMOVED WHEN UPDATING TO cudf>=24.12 #####

cdef vector[string] get_column_names(object tbl, object index):
    cdef vector[string] column_names
    if index is not False:
        if isinstance(tbl._index, cudf.core.multiindex.MultiIndex):
            for idx_name in tbl._index.names:
                column_names.push_back(str.encode(idx_name))
        else:
            if tbl._index.name is not None:
                column_names.push_back(str.encode(tbl._index.name))

    for col_name in tbl._column_names:
        column_names.push_back(str.encode(col_name))

    return column_names

cdef extern from "morpheus/objects/table_info.hpp" namespace "morpheus" nogil:


    cdef cppclass TableInfoData:
        TableInfoData()
        TableInfoData(table_view view,
                      vector[string] indices,
                      vector[string] columns)

        table_view table_view
        vector[string] index_names
        vector[string] column_names
        vector[size_type] column_indices


cdef public api:
    object make_table_from_table_with_metadata(table_with_metadata table, int index_col_count):

        schema_infos = [x.name.decode() for x in table.metadata.schema_info]
        index_names = schema_infos[0:index_col_count] if index_col_count > 0 else None
        column_names = schema_infos[index_col_count:]

        plc_table = plc_Table.from_libcudf(move(table.tbl))

        if index_names is None:
            index = None
            data = {
                col_name: ColumnBase.from_pylibcudf(col)
                for col_name, col in zip(
                    column_names, plc_table.columns()
                )
            }
        else:
            result_columns = [
                ColumnBase.from_pylibcudf(col)
                for col in plc_table.columns()
            ]
            index = cudf.Index._from_data(
                dict(
                    zip(
                        index_names,
                        result_columns[: len(index_names)],
                    )
                )
            )
            data = dict(
                zip(
                    column_names,
                    result_columns[len(index_names) :],
                )
            )
        df = cudf.DataFrame._from_data(data, index)

        # Update the struct field names after the DataFrame is created
        update_struct_field_names(df, table.metadata.schema_info)

        return df

    object make_table_from_table_info_data(TableInfoData table_info, object owner):

        cdef table_metadata tbl_meta

        num_index_cols_meta = 0
        cdef column_name_info child_info
        for i, name in enumerate(owner._column_names, num_index_cols_meta):
            child_info.name = name.encode()
            tbl_meta.schema_info.push_back(child_info)
            _set_col_children_metadata(
                owner[name]._column,
                tbl_meta.schema_info[i]
            )

        index_names = None

        if (table_info.index_names.size() > 0):
            index_names = []

            for c_name in table_info.index_names:
                name = c_name.decode()
                index_names.append(name if name != "" else None)

        column_names = []

        for c_name in table_info.column_names:
                name = c_name.decode()
                column_names.append(name if name != "" else None)


        column_indicies = []

        for c_index in table_info.column_indices:
            column_indicies.append(c_index)

        try:
            data, index = data_from_table_view_indexed(
                table_info.table_view,
                owner=owner,
                column_names=column_names,
                column_indices=column_indicies,
                index_names=index_names
            )
        except Exception:
            import traceback
            print("error while converting libcudf table to cudf dataframe:", traceback.format_exc())

        df = cudf.DataFrame._from_data(data, index)

        update_struct_field_names(df, tbl_meta.schema_info)

        return df


    TableInfoData make_table_info_data_from_table(object table):

        cdef vector[string] temp_col_names = get_column_names(table, True)

        cdef plc_Table plc_table = plc_Table(
            [
                col.to_pylibcudf(mode="read")
                for col in itertools.chain(table.index._columns, table._columns)
            ]
        )
        cdef table_view input_table_view = plc_table.view()
        cdef vector[string] index_names
        cdef vector[string] column_names

        # cuDF does a weird check where if there is only one name in both index and columns, and that column is empty or
        # None, then change it to '""'. Not sure what this is used for
        check_empty_name = get_column_names(table, True).size() == 1

        for name in table._index.names:
            if (check_empty_name and name in (None, '')):
                name = '""'
            elif (name is None):
                name = ""

            index_names.push_back(str.encode(name))

        for name in table._column_names:
            if (check_empty_name and name in (None, '')):
                name = '""'
            elif (name is None):
                name = ""

            column_names.push_back(str.encode(name))

        return TableInfoData(input_table_view, index_names, column_names)

    cdef data_from_table_view_indexed(
        table_view tv,
        object owner,
        object column_names,
        object column_indices,
        object index_names
    ):
        """
        Given a ``cudf::table_view``, constructs a Frame from it,
        along with referencing an ``owner`` Python object that owns the memory
        lifetime. If ``owner`` is a Frame we reach inside of it and
        reach inside of each ``cudf.Column`` to make the owner of each newly
        created ``Buffer`` underneath the ``cudf.Column`` objects of the
        created Frame the respective ``Buffer`` from the relevant
        ``cudf.Column`` of the ``owner`` Frame
        """
        cdef size_type column_idx = 0
        table_owner = isinstance(owner, cudf.core.frame.Frame)

        # First construct the index, if any
        index = None
        if index_names is not None:
            index_columns = []
            for _ in index_names:
                column_owner = owner
                if table_owner:
                    column_owner = owner._index._columns[column_idx]
                index_columns.append(
                    from_column_view_with_fix(
                        tv.column(column_idx),
                        column_owner
                    )
                )
                column_idx += 1
            index = cudf.core.index._index_from_data(
                dict(zip(index_names, index_columns)))

        # Construct the data dict
        cdef size_type source_column_idx = 0
        data_columns = []
        for _ in column_names:
            column_owner = owner
            if table_owner:
                column_owner = owner._columns[column_indices[source_column_idx]]
            data_columns.append(
                from_column_view_with_fix(tv.column(column_idx), column_owner)
            )
            column_idx += 1
            source_column_idx += 1

        return dict(zip(column_names, data_columns)), index

cdef _set_col_children_metadata(col,
                                column_name_info& col_meta):
    cdef column_name_info child_info
    if isinstance(col.dtype, cudf.StructDtype):
        for i, (child_col, name) in enumerate(
            zip(col.children, list(col.dtype.fields))
        ):
            child_info.name = name.encode()
            col_meta.children.push_back(child_info)
            _set_col_children_metadata(
                child_col, col_meta.children[i]
            )
    elif isinstance(col.dtype, cudf.ListDtype):
        for i, child_col in enumerate(col.children):
            col_meta.children.push_back(child_info)
            _set_col_children_metadata(
                child_col, col_meta.children[i]
            )
    else:
        return

cdef update_struct_field_names(
    table,
    vector[column_name_info]& schema_info
):
    for i, (name, col) in enumerate(table._data.items()):
        table._data[name] = update_column_struct_field_names(
            col, schema_info[i]
        )


cdef update_column_struct_field_names(
    col,
    column_name_info& info
):
    cdef vector[string] field_names

    if col.dtype != "object" and col.children:
        children = list(col.children)
        for i, child in enumerate(children):
            children[i] = update_column_struct_field_names(
                child,
                info.children[i]
            )
            col.set_base_children(tuple(children))

    if isinstance(col.dtype, StructDtype):
        field_names.reserve(len(col.base_children))
        for i in range(info.children.size()):
            field_names.push_back(info.children[i].name)
        col = col._rename_fields(
            field_names
        )

    return col
