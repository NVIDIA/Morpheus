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


class FeatureConstants():

    FILE_EXTN_EXP = '.COM;.EXE;.BAT;.CMD;.VBS;.VBE;.JS;.JSE;.WSF;.WSH;.MSC;.CPL'

    FULL_MEMORY_ADDRESS = 2147483647

    HANDLES_TYPES = [('Directory', 'directory'), ('TpWorkerFactory', 'tpworkerfactory'),
                     ('WaitCompletionPacket', 'waitcompletionpacket'), ('Section', 'section'), ('File', 'file'),
                     ('Mutant', 'mutant'), ('Event', 'event'), ('Semaphore', 'semaphore'), ('Key', 'key'),
                     ('IoCompletion', 'iocompletion'), ('ALPC Port', 'alpc port'), ('Thread', 'thread')]

    HANDLES_TYPES_2 = [('IoCompletionReserve', 'iocompletionreserve'), ('Desktop', 'desktop'),
                       ('EtwRegistration', 'etwregistration'), ('WindowStation', 'windowstation')]

    PROTECTIONS = {
        'PAGE_EXECUTE_READWRITE ': 'page_execute_readwrite',
        'PAGE_NOACCESS ': 'page_noaccess',
        'PAGE_EXECUTE_WRITECOPY ': 'page_execute_writecopy',
        'PAGE_READONLY ': 'page_readonly',
        'PAGE_READWRITE ': 'page_readwrite'
    }

    WAIT_REASON_LIST = ['9', '31', '13']

    VAD = 'Vad '

    VADS = 'VadS'

    PAGE_NOACCESS = 'PAGE_NOACCESS '

    PAGE_EXECUTE_READWRITE = 'PAGE_EXECUTE_READWRITE '

    PAGE_EXECUTE_WRITECOPY = 'PAGE_EXECUTE_WRITECOPY '

    PAGE_READONLY = 'PAGE_READONLY '

    PAGE_READWRITE = 'PAGE_READWRITE '
