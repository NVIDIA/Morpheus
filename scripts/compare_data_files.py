#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import sys

from morpheus.common import FileTypes
from morpheus.io.deserializers import read_file_to_df
from morpheus.utils.compare_df import compare_df
from morpheus.utils.logger import configure_logging


def parse_args():
    argparser = argparse.ArgumentParser("Compares two data files which are parsable as Pandas dataframes")
    argparser.add_argument("data_files", nargs=2, help="Files to compare")
    argparser.add_argument('--include',
                           nargs='*',
                           help=("Which columns to include in the validation. "
                                 "Resulting columns is the intersection of all regex. Include applied before exclude"))
    argparser.add_argument(
        '--exclude',
        nargs='*',
        default=[r'^ID$', r'^_ts_'],
        help=("Which columns to exclude from the validation. "
              "Resulting ignored columns is the intersection of all regex. Include applied before exclude"))
    argparser.add_argument(
        '--index_col',
        help=("Specifies a column which will be used to align messages with rows in the validation dataset."))
    argparser.add_argument('--abs_tol',
                           type=float,
                           default=0.001,
                           help="Absolute tolerance to use when comparing float columns.")
    argparser.add_argument('--rel_tol',
                           type=float,
                           default=0.05,
                           help="Relative tolerance to use when comparing float columns.")
    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    configure_logging(log_level=logging.DEBUG)

    df_a = read_file_to_df(args.data_files[0], file_type=FileTypes.Auto, df_type='pandas')
    df_b = read_file_to_df(args.data_files[1], file_type=FileTypes.Auto, df_type='pandas')
    results = compare_df(df_a,
                         df_b,
                         include_columns=args.include,
                         exclude_columns=args.exclude,
                         replace_idx=args.index_col,
                         abs_tol=args.abs_tol,
                         rel_tol=args.rel_tol)

    if results['diff_rows'] > 0 or results['diff_cols'] > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
