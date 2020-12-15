# -*- coding: utf-8 -*-

# Copyright 2020 Google LLC
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
#
from google.cloud.aiplatform_helpers.value_converter import to_value
from google.cloud.aiplatform_helpers.value_converter import from_value
from google.cloud.aiplatform_helpers.value_converter import from_map

from proto.marshal import Marshal
from proto.marshal.rules.struct import ValueRule
from google.protobuf.struct_pb2 import Value


class ConversionValueRule(ValueRule):
    def to_python(self, value, *, absent: bool = None):
        return super().to_python(value, absent=absent)

    def to_proto(self, value):

        # Need to check whether value is an instance
        # of an enhanced type
        if callable(getattr(value, "to_value", None)):
            return value.to_value()
        else:
            return super().to_proto(value)


def add_methods_to_classes_in_package(pkg):
    classes = dict(
        [(name, cls) for name, cls in pkg.__dict__.items() if isinstance(cls, type)]
    )

    for class_name, cls in classes.items():
        # Add to_value() method to class with docstring
        setattr(cls, "to_value", to_value)
        cls.to_value.__doc__ = to_value.__doc__

        # Add from_value() method to class with docstring
        cls.from_value = add_from_value_to_class(cls)
        cls.from_value.__doc__ = from_value.__doc__

        # Add from_map() method to class with docstring
        setattr(cls, "from_map", add_from_map_to_class(cls))
        cls.from_map.__doc__ = from_map.__doc__


def add_from_value_to_class(cls):
    def _from_value(value):
        return from_value(cls, value)

    return _from_value


def add_from_map_to_class(cls):
    def _from_map(map_):
        return from_map(cls, map_)

    return _from_map


marshal = Marshal(name="google.cloud.aiplatform.v1beta1")
marshal.register(Value, ConversionValueRule(marshal=marshal))
