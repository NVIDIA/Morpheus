# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

from pylint.checkers import BaseChecker


class MorpheusLibImportChecker(BaseChecker):
    """
    Custom pylint checker to detect incorrect imports of morpheus._lib modules.

    refer to https://pylint.readthedocs.io/en/stable/development_guide/how_tos/custom_checkers.html
    """
    name = 'morpheus_lib_import_checker'
    msgs = {
        'W9901': ('Incorrect import of %(name)s as %(alias)s. Imports from morpheus._lib should be in the form of: '
                  '"import morpheus._lib.%(name)s as %(expected_alias)s"',
                  'morpheus-incorrect-lib-import',
                  'Used when a forbidden library is imported.'),
        'W9902':
            ('Incorrect import of %(name)s from %(modname)s as %(alias)s. Importing symbols from morpheus._lib should '
             'only occur from an __init__.py file: '
             '"from morpheus.%(short_mod)s import %(name)s"',
             'morpheus-incorrect-lib-from-import',
             'Used when a forbidden library is imported.'),
    }

    _LIB_MODULES = ['morpheus._lib']

    def visit_import(self, node) -> None:
        """
        Bans the following:
        * import morpheus._lib.XXX
        * import morpheus._lib.XXX as YYY
        """
        for name, alias in node.names:
            for lib_module in self._LIB_MODULES:
                if name.startswith(lib_module):
                    expected_alias = f"_{name.split('.')[-1]}"
                    if alias != expected_alias:
                        self.add_message('morpheus-incorrect-lib-import',
                                         node=node,
                                         args={
                                             'name': name, 'alias': alias, 'expected_alias': expected_alias
                                         })

    def visit_importfrom(self, node) -> None:
        """
        Bans: from morpheus._lib.XXX import YYY
        """
        for name, alias in node.names:
            for lib_module in self._LIB_MODULES:
                if (node.modname.startswith(lib_module) and os.path.basename(node.root().file) != '__init__.py'):
                    self.add_message('morpheus-incorrect-lib-from-import',
                                     node=node,
                                     args={
                                         'name': name,
                                         'alias': alias,
                                         'modname': node.modname,
                                         'short_mod': node.modname.split('.')[-1]
                                     })


def register(linter):
    linter.register_checker(MorpheusLibImportChecker(linter))
