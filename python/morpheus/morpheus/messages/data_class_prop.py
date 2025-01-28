# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing


class DataClassProp:
    """
    This class is used to configure dataclass fields within message container classes.

    Parameters
    ----------
    fget : typing.Callable[[typing.Any, str], typing.Any], optional
        Callable for field getter, by default None.
    fset : typing.Callable[[typing.Any, str, typing.Any], None], optional
        Callable for field setter, by default None.
    fdel : typing.Callable[[typing.Any, str], typing.Any], optional
        This is not used, by default None.
    doc : _type_, optional
        Documentation for field, by default None.
    field : _type_, optional
        Field value, by default None.
    """

    def __init__(self,
                 fget: typing.Callable[[typing.Any, str], typing.Any] = None,
                 fset: typing.Callable[[typing.Any, str, typing.Any], None] = None,
                 fdel=None,
                 doc=None,
                 field=None):

        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.name = None
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc
        self._field = field

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if (instance is None):
            # Most likely, this is getting the default field value for the dataclass.
            return self._field

        if self.fget is None:
            raise AttributeError("unreadable attribute")

        return self.fget(instance, self.name)

    def __set__(self, instance, value):

        if (instance is None):
            return

        if self.fset is None:
            raise AttributeError("can't set attribute")

        self.fset(instance, self.name, value)

    def __delete__(self, instance):
        if (instance is None):
            return

        del instance.inputs[self.name]
