# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

import typing

import pandas as pd

from .data_models import FeatureConfig
from .data_models import ProtectionData
from .feature_constants import FeatureConstants as fc


class FeatureExtractor():
    """
    This is a helper class to extract required features for ransomware detection pipeline.
    """

    def __init__(self, config: FeatureConfig) -> None:
        self._config = config
        self._features = None

    def _filter_by_pid_process(self, plugin_dict: typing.Dict[str, pd.DataFrame],
                               pid_process: str) -> typing.Dict[str, pd.DataFrame]:
        """
        This function filter plugins data by pid_process.
        """

        filtered_plugin_dict = {}

        for plugin_name in plugin_dict.keys():
            plugin_df = plugin_dict[plugin_name]
            plugin_df = plugin_df[plugin_df.PID_Process == pid_process]
            filtered_plugin_dict[plugin_name] = plugin_df

        return filtered_plugin_dict

    def _count_double_extension(self, file_paths: typing.List[str]):
        """
        This function counts the amount of double extensions to a common type files and
        return the largest double extension.
        """

        count = 0
        max_ext_word_dot = 0

        for file_path in file_paths:

            file_split_dot = file_path.split('.')

            split_dot = file_split_dot[:-1]

            if len(split_dot) > 1:

                for word_dot in split_dot:

                    if word_dot in self._config.file_extns:
                        count += 1
                        index_word_dot = file_split_dot.index(word_dot)
                        ext_word_dot = ".".join(file_split_dot[index_word_dot + 1:])
                        if len(ext_word_dot) > max_ext_word_dot:
                            max_ext_word_dot = len(ext_word_dot)

                        break

        self._features['count_double_extension_count_handles'] = count
        self._features['double_extension_len_handles'] = max_ext_word_dot

    def _extract_envars(self, x: pd.DataFrame):
        """
        This function extracts environment features.
        """

        x = x[x.Variable.str.contains('PATHEXT', regex=False)]
        x = x[x.Value.str.contains(fc.FILE_EXTN_EXP, regex=False)]

        if not x.empty:
            self._features['envirs_pathext'] = 1

        self._features['envars_df_count'] = len(x)

    def _extract_threadlist(self, x: pd.DataFrame):
        """
        # Count amount of unique states and wait reasons and thread with state and waitreason:
        # '2'-'Running'
        # '9'-'WrPageIn'
        # '13'-'WrUserRequest'
        # '31'-'WrDispatchInt'
        """

        x_state2 = x[x.State == '2']
        x_state_unique = x.State.unique()
        x_waitreason_unique = x.WaitReason.unique()

        self._features['threadlist_df_count'] = len(x)
        self._features['threadlist_df_state_2'] = len(x_state2)
        self._features['threadlist_df_state_unique'] = len(x_state_unique)
        self._features['threadlist_df_wait_reason_unique'] = len(x_waitreason_unique)

        for wait_reason in fc.WAIT_REASON_LIST:
            wait_reason_df = x[x.WaitReason == wait_reason]
            self._features['threadlist_df_wait_reason_' + wait_reason] = len(wait_reason_df)

    def _extract_vad_cc(self, commit_charge: pd.Series):
        """
        This function extracts 'vad' specific commit charge features.
        """

        cc_size = len(commit_charge)

        # Calculate mean, max, sum of commit charged of vad
        if cc_size:
            self._features['get_commit_charge_mean_vad'] = commit_charge.mean()
            self._features['get_commit_charge_max_vad'] = commit_charge.max()
            self._features['get_commit_charge_sum_vad'] = commit_charge.sum()

    def _extract_cc(self, commit_charge: pd.Series):
        """
        This function extracts commit charge features.
        """

        cc_size = len(commit_charge)

        # Calculate mean, max, sum, len of the commit charged
        if cc_size:
            self._features['get_commit_charge_mean'] = commit_charge.mean()
            self._features['get_commit_charge_max'] = commit_charge.max()
            self._features['get_commit_charge_sum'] = commit_charge.sum()
            self._features['get_commit_charge_len'] = cc_size

    def _extract_vads_cc(self, commit_charge: pd.Series, vads_cc: pd.Series):
        """
        This function extracts 'vads' commit charge features.
        """

        cc_size = len(commit_charge)

        # Calculate min of commit charged of vads
        if cc_size:
            self._features['get_commit_charge_min_vads'] = commit_charge.min()

        # Calculate the amount of entire memory commit charged of vads
        commit_charge = vads_cc[vads_cc == fc.FULL_MEMORY_ADDRESS]
        self._features['count_entire_commit_charge_vads'] = len(commit_charge)

    def _extract_cc_vad_page_noaccess(self, commit_charge: pd.Series):
        """
        This function extracts 'vad' commit charge features specific to 'page_noaccess' protection.
        """

        commit_charge = commit_charge[commit_charge < fc.FULL_MEMORY_ADDRESS]

        # Calculate min and mean of commit charged of vad memory with PAGE_NOACCESS protection
        if not commit_charge.empty:
            self._features['get_commit_charge_min_vad_page_noaccess'] = commit_charge.min()
            self._features['get_commit_charge_mean_vad_page_noaccess'] = commit_charge.mean()

    def _extract_unique_file_extns(self, x: pd.DataFrame):
        """
        This function extracts unique file extenstion featurs.
        """

        vadinfo_files = x[x.File != 'N/A'].File

        # Count the amount of unique file extensions
        if not vadinfo_files.empty:
            unique_file_extns = vadinfo_files.str.lower().str.extract('(\\.[^.]*)$')[0].dropna().unique()
            self._features['get_count_unique_extensions'] = len(unique_file_extns)

    def _extract_vadinfo(self, x: pd.DataFrame):
        """
        This function extracts vadinfo features about commit charged, vad/vads and
        private memory and memory protection type.
        """

        # CommitCharge - is the total amount of virtual memory of all processes that
        # must be backed by either physical memory or the page file
        # vad - virtual address descriptor
        # vads - virtual address descriptor short
        # private memory - this field refers to committed regions that cannot be shared with other processes.

        # Calculate the commit charges of vad and vads
        vad_cc = x[x.Tag == fc.VAD].CommitCharge
        vads_cc = x[x.Tag == fc.VADS].CommitCharge

        vadinfo_size = len(vad_cc)
        vadsinfo_size = len(vads_cc)

        vad_size = len(x)

        private_memory_one_df = x[x.PrivateMemory == '1']
        vad_private_memory_len = len(private_memory_one_df)

        # Count vad, vads and private memory amount
        self._features['vad_count'] = vadinfo_size
        self._features['vads_count'] = vadsinfo_size
        self._features['count_private_memory'] = len(private_memory_one_df)

        # Calculate the ratio of vad and private memory in reduce time delay bias
        if vad_size:
            self._features['ratio_private_memory'] = (vad_private_memory_len / vad_size)
            self._features['vad_ratio'] = (vadinfo_size / vad_size)

        commit_charge = x[x.CommitCharge < fc.FULL_MEMORY_ADDRESS].CommitCharge
        self._extract_cc(commit_charge)

        # calculating the amount of commit charged of vad
        commit_charge = vad_cc[vad_cc < fc.FULL_MEMORY_ADDRESS]
        self._extract_vad_cc(commit_charge)

        # Calculate the amount of commit charged of vads
        commit_charge = vads_cc[vads_cc < fc.FULL_MEMORY_ADDRESS]
        self._extract_vads_cc(commit_charge, vads_cc)

        # calculating commit charged of memory with PAGE_NOACCESS protection
        commit_charge = x[(x.Protection == fc.PAGE_NOACCESS) & (x.Tag == fc.VAD)].CommitCharge
        self._extract_cc_vad_page_noaccess(commit_charge)

        self._extract_protections(x, vad_size, vadsinfo_size, vadinfo_size)

        self._extract_unique_file_extns(x)

    def _get_protection_data(self,
                             x: pd.DataFrame,
                             protection: str,
                             vadinfo_df_size: int,
                             vadsinfo_size: int,
                             vadinfo_size: int):
        """
        This function creates protection data instance.
        """

        protection_df = x[x.Protection == protection]
        commit_charge = protection_df.CommitCharge
        commit_charge = commit_charge[commit_charge < fc.FULL_MEMORY_ADDRESS]
        vads_protection_size = len(protection_df[protection_df.Tag == fc.VADS])
        vad_protection_size = len(protection_df[protection_df.Tag == fc.VAD])
        commit_charge_size = len(commit_charge)
        protection_df_size = len(protection_df)
        protection_id = fc.PROTECTIONS[protection]

        p_data = ProtectionData(commit_charge,
                                vads_protection_size,
                                vad_protection_size,
                                commit_charge_size,
                                protection_df_size,
                                protection_id,
                                vadinfo_df_size,
                                vadsinfo_size,
                                vadinfo_size)

        return p_data

    def _page_execute_readwrite(self, x: ProtectionData):
        """
        This function extracts 'page_execute_readwrite' protection reelated features.
        """

        commit_charge = x.commit_charges

        if x.commit_charge_size:
            self._features['get_commit_charge_mean_page_execute_readwrite'] = commit_charge.mean()
            self._features['get_commit_charge_min_page_execute_readwrite'] = commit_charge.min()
            self._features['get_commit_charge_max_page_execute_readwrite'] = commit_charge.max()
            self._features['get_commit_charge_sum_page_execute_readwrite'] = commit_charge.sum()
            self._features['get_commit_charge_std_page_execute_readwrite'] = commit_charge.std(ddof=0)

        # Calculate amount and ratio of memory pages with 'PAGE_EXECUTE_READWRITE protection
        if x.protection_df_size:
            self._features['page_execute_readwrite_count'] = x.protection_df_size
            self._features['page_execute_readwrite_ratio'] = (x.protection_df_size / x.vadinfo_df_size)

        if x.vads_protection_size:
            # Calculate amount and ratio of vads memory pages with 'PAGE_EXECUTE_READWRITE' protection
            self._features['page_execute_readwrite_vads_count'] = x.vads_protection_size
            self._features['page_execute_readwrite_vads_ratio'] = (x.vads_protection_size / x.vadsinfo_size)

    def _page_noaccess(self, x: ProtectionData):
        """
        This function extracts 'page_noaccess' protection reelated features.
        """

        commit_charge = x.commit_charges

        if x.commit_charge_size:
            self._features['get_commit_charge_mean_page_no_access'] = commit_charge.mean()
            self._features['get_commit_charge_min_page_no_access'] = commit_charge.min()
            self._features['get_commit_charge_max_page_no_access'] = commit_charge.max()
            self._features['get_commit_charge_sum_page_no_access'] = commit_charge.sum()

        # Calculate amount and ratio of memory pages with 'PAGE_NOACCESS' protection
        if x.protection_df_size:
            self._features['page_no_access_count'] = x.protection_df_size
            self._features['page_no_access_ratio'] = (x.protection_df_size / x.vadinfo_df_size)

        # Calculate amount and ratio of vad and vads memory pages with 'PAGE_NOACCESS' protection
        self._features['page_no_access_vads_count'] = x.vads_protection_size
        self._features['page_no_access_vad_count'] = x.vad_protection_size

        if x.vads_protection_size:
            self._features['page_no_access_vads_ratio'] = (x.vads_protection_size / x.vadsinfo_size)

        if x.vad_protection_size:
            self._features['page_no_access_vad_ratio'] = (x.vad_protection_size / x.vadinfo_size)

    def _page_execute_writecopy(self, x: ProtectionData):
        """
        This function extracts 'page_execute_writecopy' protection reelated features.
        """

        commit_charge = x.commit_charges

        # Calculate min and sum of commit charged with memory pages with 'PAGE_EXECUTE_WRITECOPY' protection
        if x.commit_charge_size:
            self._features['get_commit_charge_min_page_execute_writecopy'] = commit_charge.min()
            self._features['get_commit_charge_sum_page_execute_writecopy'] = commit_charge.sum()

        # Calculate amount and ratio of vad memory pages with 'PAGE_EXECUTE_WRITECOPY' protection
        self._features['page_execute_writecopy_vad_count'] = x.vad_protection_size
        if x.vad_protection_size:
            self._features['page_execute_writecopy_vad_ratio'] = (x.vad_protection_size / x.vadinfo_size)

    def _page_readonly(self, x: ProtectionData):
        """
        This function extracts 'page_readonly' protection reelated features.
        """

        commit_charge = x.commit_charges

        # Calculate mean of commit charged with memory pages with 'PAGE_READONLY' protection
        if x.commit_charge_size:
            self._features['get_commit_charge_mean_page_readonly'] = commit_charge.mean()

        # Calculate amount and ratio of memory pages with 'PAGE_READONLY' protection
        if x.protection_df_size:
            self._features['page_readonly_count'] = x.protection_df_size
            self._features['page_readonly_ratio'] = (x.protection_df_size / x.vadinfo_df_size)

        # Calculate amount and ratio of vad and vads memory pages with 'PAGE_READONLY' protection
        self._features['page_readonly_vads_count'] = x.vads_protection_size
        self._features['page_readonly_vad_count'] = x.vad_protection_size

        if x.vads_protection_size:
            self._features['page_readonly_vads_ratio'] = (x.vads_protection_size / x.vadsinfo_size)

        if x.vad_protection_size:
            self._features['page_readonly_vad_ratio'] = (x.vad_protection_size / x.vadinfo_size)

    def _page_readwrite(self, x: ProtectionData):
        """
        This function extracts 'page_readwrite' protection reelated features.
        """

        # Calculate ratio of memory pages with 'PAGE_READWRITE' protection
        if x.protection_df_size:
            self._features['page_readwrite_ratio'] = (x.protection_df_size / x.vadinfo_df_size)

        # Calculate amount and ratio of vad and vads memory pages with 'PAGE_READWRITE' protection
        self._features['page_readwrite_vads_count'] = x.vads_protection_size
        self._features['page_readwrite_vad_count'] = x.vad_protection_size

        if x.vads_protection_size:
            self._features['page_readwrite_vads_ratio'] = (x.vads_protection_size / x.vadsinfo_size)

        if x.vad_protection_size:
            self._features['page_readwrite_vad_ratio'] = (x.vad_protection_size / x.vadinfo_size)

    def _extract_protections(self, x: pd.DataFrame, vadinfo_df_size: int, vadsinfo_size: int, vadinfo_size: int):
        """
        This function extracts protection features related to vadinfo plugin.
        """
        page_execute_writecopy_count = 0

        for protection in fc.PROTECTIONS:

            p_data = self._get_protection_data(x, protection, vadinfo_df_size, vadsinfo_size, vadinfo_size)

            # Calculate features related to memory pages with 'PAGE_EXECUTE_READWRITE' access
            if protection == fc.PAGE_EXECUTE_READWRITE:
                # Calculate mean, min, max, sum and std of commit charged with memory pages with 'PAGE_EXECUTE_READWRITE
                # protection
                self._page_execute_readwrite(p_data)

            # Calculate features related to memory pages with 'PAGE_NOACCESS' access
            elif protection == fc.PAGE_NOACCESS:
                # Calculate mean, min, max and sum of commit charged with memory pages with 'PAGE_NOACCESS'
                # protection
                self._page_noaccess(p_data)

            # Calculate features related to memory pages with 'PAGE_EXECUTE_WRITECOPY' access
            elif protection == fc.PAGE_EXECUTE_WRITECOPY:
                self._page_execute_writecopy(p_data)
                page_execute_writecopy_count = p_data.protection_df_size

            # Calculate features related to memory pages with 'PAGE_READONLY' access
            elif protection == fc.PAGE_READONLY:
                self._page_readonly(p_data)

            # Calculate features related to memory pages with 'PAGE_READWRITE' access
            elif protection == fc.PAGE_READWRITE:
                self._page_readwrite(p_data)

            else:
                continue

        # Count the amount of unique file paths in vadinfo
        self._features['vadinfo_df_path_unique'] = len(x.File.unique())
        self._features['vads_page_execute_writecopy_ratio'] = vadsinfo_size / (page_execute_writecopy_count + 1)

    def _extract_handle_types(self, x: pd.DataFrame):
        """
        This function extracts file handle type features from handles plugin.
        """

        # Get count and ratio for the handles by their type.
        for h_type in (fc.HANDLES_TYPES + fc.HANDLES_TYPES_2):

            df = x[x.Type == h_type[0]]
            df_len = len(df)

            if h_type in fc.HANDLES_TYPES:
                col = 'handles_df_' + h_type[1] + '_count'
                self._features[col] = df_len

            col = 'handles_df_' + h_type[1] + '_ratio'
            self._features[col] = df_len / (self._features['handles_df_count'] + 1)

    def _extract_file_handle_dirs(self, file_paths: pd.Series):
        """
        This function extracts file handle directory features from handles plugin.
        """

        filepath_split_df = file_paths[file_paths.str.split('\\').str.len() > 3].str.split('\\', expand=True)

        if not filepath_split_df.empty:
            # Count the unique directories
            directories_uniques_count = len(file_paths.str.extract('^(.*)\\\\.*')[0].unique())
            if len(filepath_split_df) > 3:
                filepath_split_df = filepath_split_df[(~filepath_split_df[4].isna())
                                                      & (filepath_split_df[1] == 'device') &
                                                      (filepath_split_df[2].str.contains('harddisk'))]

                windows = filepath_split_df[filepath_split_df[3].str.contains('windows')]
                users = filepath_split_df[filepath_split_df[3].str.contains('users')]

                # Count handles files of personal users directories
                self._features['file_users_exists'] = len(users)

                # Count handles files of Windows directories
                self._features['file_windows_count'] = len(windows)

            # Count amount of unique directories
            self._features['count_directories_handles_uniques'] = directories_uniques_count

    def _extract_handles(self, x: pd.DataFrame):
        """
        This function extracts features related to handles such as amount and ratio of each handle type.
        """

        # Amount of files path in handles files
        file_paths = x[x.Type == 'File'].Name.str.lower()

        file_paths = file_paths[(~file_paths.isna()) & (file_paths != '')]

        # Count handles files with double extensions
        self._count_double_extension(file_paths=list(file_paths))

        # Count handles files with common extension
        file_extensions = file_paths.str.extract('\\.([^.]*)$')[0].dropna()

        file_extns = file_extensions[file_extensions.isin(self._config.file_extns)]
        self._features['check_doc_file_handle_count'] = len(file_extns)

        self._extract_file_handle_dirs(file_paths)

        # Count unique file handles extensions
        self._features['count_extension_handles_uniques'] = len(file_extensions.unique())

        # Count number of handles
        self._features['handles_df_count'] = len(x)
        name_unique_count = len(x.Name.unique())

        # Count handles with unique name
        self._features['handles_df_name_unique'] = name_unique_count

        # Calculate the ratio of handles with unique name
        self._features['handles_df_name_unique_ratio'] = name_unique_count / (self._features['handles_df_count'] + 1)

        type_unique_count = len(x.Type.unique())

        # Count the amount of unique handles type
        self._features['handles_df_type_unique'] = type_unique_count

        # Calculate the ratio of handles with unique type
        self._features['handles_df_type_unique_ratio'] = type_unique_count / (self._features['handles_df_count'] + 1)

        self._extract_handle_types(x)

    def _extract_ldrmodules(self, x: pd.DataFrame):
        """
        This function extracts size of the ldrmodules process and it's path.
        """

        if not x.empty:
            process = x.Process.iloc[0].lower()
            x = x[x.Name.str.contains(process)]
            if not x.empty:
                self._features['ldrmodules_df_size_int'] = int(x.Size.iloc[0], 16)
                self._features['ldrmodules_df_path'] = x.Path.iloc[0]
            else:
                self._features['ldrmodules_df_path'] = ""

    def extract_features(self, x: pd.DataFrame, feas_all_zeros: typing.Dict[str, int]) -> pd.DataFrame:
        """
        This function extracts all different ransomware features.

        Parameters
        ----------
        x : `pandas.DataFrame`
            Dataframe with appshield snapshot data.
        feas_all_zeros : typing.Dict[str, int]
            Features with default value (0)
        Returns
        -------
        pandas.DataFrame
            Ransomware features dataframe.
        """

        features_per_pid_process = []

        # Get unique PID_Process for a given snapshot
        pid_processes = list(x["PID_Process"].unique())

        # Filter snapshot data by plugin
        plugin_dict = {plugin: x[x.plugin == plugin] for plugin in self._config.interested_plugins}

        # Filter plugin per pid_process and create features
        for pid_process in pid_processes:

            # Setting default value '0' to all features.
            self._features = feas_all_zeros.copy()

            fltr_plugin_dict = self._filter_by_pid_process(plugin_dict, pid_process)

            try:
                ldrmodules_df = fltr_plugin_dict['ldrmodules']
                threadlist_df = fltr_plugin_dict['threadlist']
                envars_df = fltr_plugin_dict['envars']
                vadinfo_df = fltr_plugin_dict['vadinfo']
                handles_df = fltr_plugin_dict['handles']

            except KeyError as e:
                raise KeyError(f'Missing required plugins: {e}') from e

            # Envars plugin features displays a process's environment variables.
            # Typically this will show the number of CPUs installed and the hardware architecture,
            # the process's current directory, temporary directory, session name, computer name, user name,
            # and various other interesting artifacts.
            self._extract_envars(envars_df)

            # Threadlist plugin features displays the threads that are used by a process.
            self._extract_threadlist(threadlist_df)

            # VadInfo plugin features displays extended information about a process's VAD nodes.
            # In particular, it shows:
            # - The address of the MMVAD structure in kernel memory
            # - The starting and ending virtual addresses in process memory that the MMVAD structure pertains to
            # - The VAD Tag
            # - The VAD flags, control flags, etc
            # - The name of the memory mapped file (if one exists)
            # - The memory protection constant (permissions)
            self._extract_vadinfo(vadinfo_df)

            # Handles plugin features displays the open handles in a process, use the handles command.
            # This applies to files, registry keys, mutexes, named pipes, events, window stations, desktops, threads,
            # and all other types of securable executive objects.
            self._extract_handles(handles_df)

            # LdrModules plugin features displays a process's loaded DLLs. LdrModules detects a dll-hiding or injection
            # kind of activities in a process memory.
            self._extract_ldrmodules(ldrmodules_df)

            # Add pid_process
            self._features['pid_process'] = pid_process

            # Add pid_process features to a list
            features_per_pid_process.append(self._features)

        # Convert list of pid_process features to a dataframe
        features_df = pd.DataFrame.from_dict(features_per_pid_process)

        # Snapshot id is used to determine which snapshot the pid_process belongs to
        features_df['snapshot_id'] = x.snapshot_id.iloc[0]

        # Add timestamp. Here we consider only ldrmodules timestamp for all the entries.
        features_df['timestamp'] = plugin_dict['ldrmodules'].timestamp.iloc[0]

        return features_df

    @staticmethod
    def combine_features(x: typing.List[pd.DataFrame]) -> pd.DataFrame:
        """
        This function combines features of multiple snapshots to a single dataframe

        Parameters
        ----------
        x : `typing.List[pd.DataFrame]`
            Features of multiple snapshots.

        Returns
        -------
        pandas.DataFrame
            Ransomware features dataframe.
        """
        return pd.concat(x)
