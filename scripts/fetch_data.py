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
import os
import subprocess
import sys
import time

LFS_DATASETS = {
    'all': '**',
    'datasets': 'models/datasets/**',
    'docs': 'docs/**',
    'examples': 'examples/**',
    'models': 'models/**',
    'tests': 'tests/**',
    'validation': 'models/datasets/validation-data/**'
}


def lfs_pull(include_paths, poll_interval=0.1):
    """
    Performs a git lfs pull.
    """
    include_paths = ','.join(include_paths)
    cmd = f'git lfs pull -X "" -I "{include_paths}"'
    env = os.environ.copy()

    # Instruct git lfs to not supress progress output. Fetching the models can
    # take over a minute to complete, so we want our users to receive feedback.
    env['GIT_LFS_FORCE_PROGRESS'] = '1'

    with subprocess.Popen(cmd,
                          env=env,
                          shell=True,
                          universal_newlines=True,
                          stderr=subprocess.STDOUT,
                          stdout=subprocess.PIPE) as popen:
        outpipe = popen.stdout
        returncode = None
        all_out = []
        while returncode is None:
            time.sleep(poll_interval)
            out = outpipe.readline()
            if out.rstrip() != '':
                logging.info(out.rstrip())
                all_out.append(out)

            returncode = popen.poll()

        # Check if we have any additional output written to the pipe before our last poll
        out = outpipe.read()
        if out != '':
            all_out.append(out)

        output = ''.join(all_out).rstrip("\n")
        if returncode != 0:
            logging.error(output)
            raise subprocess.CalledProcessError(returncode=returncode, cmd=cmd, output=output)

        return output


def lfs_check(list_all=False):
    output = subprocess.check_output('git lfs ls-files', shell=True, universal_newlines=True)
    output_lines = output.splitlines()

    # Output lines are in the format of:
    # <oid> [-|*] <file name>
    # where '-' indicates a file pointer and '*' indicates a downloaded file
    # https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-ls-files.1.ronn
    missing_files = []
    for file_status in output_lines:
        parts = file_status.split()
        downloaded = parts[1] == '*'
        filename = ' '.join(parts[2:])

        if not downloaded:
            missing_files.append(filename)

        if list_all:
            # the join on 2: is needed to handle file names that contain a blank space
            logging.info('%s - %s', filename, downloaded)

    if not list_all:
        if missing_files:
            logging.error("Missing the following LFS files:\n%s", "\n".join(missing_files))
        else:
            logging.info("All LFS files downloaded")

    if missing_files:
        sys.exit(1)


def parse_args():
    argparser = argparse.ArgumentParser("Fetches data not included in the repository by default")
    subparsers = argparser.add_subparsers(title='Subcommands',
                                          description='valid subcommands',
                                          required=True,
                                          dest='subcommand')

    fetch_parser = subparsers.add_parser('fetch', help='Fetch datasets')
    fetch_parser.add_argument("data_set", nargs='*', choices=list(LFS_DATASETS.keys()), help="Data set to fetch")

    check_parser = subparsers.add_parser('check',
                                         help=('Check download status of large files. Exits with a status of 0 if all '
                                               'large files have been downloaded, 1 otherwise.'))
    check_parser.add_argument("-l",
                              "--list",
                              action="store_true",
                              default=False,
                              dest='list_all',
                              help="List all missing files")

    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.subcommand == 'fetch':
        include_paths = [LFS_DATASETS[p] for p in args.data_set]
        lfs_pull(include_paths)
    else:
        lfs_check(list_all=args.list_all)


if __name__ == "__main__":
    main()
