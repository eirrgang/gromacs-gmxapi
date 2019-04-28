#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright (c) 2019, by the GROMACS development team, led by
# Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
# and including many others, as listed in the AUTHORS file in the
# top-level source directory and at http://www.gromacs.org.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at http://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out http://www.gromacs.org.

"""gmxapi data types and interfaces.

The data types defined here are Abstract Data Types in the styles of both
collections.abc and typing modules. They can be used to validate interfaces or
data compatibility. They can be used as base classes for concrete types, but
code should not assume that gmxapi data objects actually inherit from these
base classes. In fact, gmxapi data objects will generally be C (or C++) objects
accompanied by C struct descriptors. ABI compatibility relies on these C structs
and the accompanying Python Capsule schema.
"""

# __all__ = ['ndarray']

import collections

from gmxapi import exceptions


# class GmxapiDataType(Enum):
#     Boolean = auto()
#     String = auto()
#     Integer64 = auto()
#     Float64 = auto()
#     AssociativeArray = auto()
#     NDArray = auto()

#
# class GmxapiDataHandle(abc.ABC):
#     """gmxapi abstract base class for a handle to an instance of data."""
#
#     @property
#     @abc.abstractmethod
#     def data_description(self):
#         return None
#
#     @property
#     @abc.abstractmethod
#     def data(self):
#         return None
#
#     @classmethod
#     def __subclasshook__(cls, C):
#         """Check if a class implements the GmxapiType interface."""
#         if cls is GmxapiDataHandle:
#             if any('data' in B.__dict__ for B in C.__mro__)\
#                     and any('data_description' in B.__dict__ for B in C.__mro__):
#                 return True
#         return NotImplemented
#
#
# class Boolean(GmxapiDataHandle):
#     """gmxapi logical value type."""
#     def __init__(self, value):
#         assert isinstance(value, bool)
#         self._data = bool(value)
#         self._data_description = DataDescription(name='', dtype=Boolean, shape=(1,))
#
#     @property
#     def data(self):
#         return self._data
#
#     @property
#     def data_description(self):
#         return self._data_description
#
#
# class String(collections.UserString, GmxapiDataHandle):
#     """gmxapi string data type."""
#     def __init__(self, string):
#         if isinstance(string, bytes):
#             super().__init__(string.decode('utf-8'))
#         elif string is None:
#             super().__init__('')
#         else:
#             super().__init__(string)
#         self._data_description = DataDescription(name='', dtype=String, shape=(1,))
#
#     @property
#     def data_description(self):
#         return self._data_description
#
#     @property
#     def data(self):
#         return super().data
#
#
# class Integer64(GmxapiDataHandle):
#     """gmxapi 64-bit integer data type."""
#     def __init__(self, number):
#         assert isinstance(number, int)
#         self._data = int(number)
#         self._data_description = DataDescription(name='', dtype=Integer64, shape=(1,))
#
#     @property
#     def data(self):
#         return self._data
#
#     @property
#     def data_description(self):
#         return self._data_description
#
#
# class Float64(GmxapiDataHandle):
#     """gmxapi 64-bit floating point data type."""
#     def __init__(self, number):
#         assert isinstance(number, float)
#         self._data = float(number)
#         self._data_description = DataDescription(name='', dtype=Float64, shape=(1,))
#
#     @property
#     def data(self):
#         return self._data
#
#     @property
#     def data_description(self):
#         return self._data_description


class NDArray(object):
    """N-Dimensional array type.

    TODO: Provide __class_getitem__ for subscripted type specification?
    TODO: Provide gmxapi Any type for TypeVar type placeholders? (ref typing module)
    """

    def __init__(self, data=None):
        self.values = []
        self.dtype = None
        self.shape = ()
        if data is not None:
            if isinstance(data, (str, bytes)):
                data = [data]
                length = 1
            else:
                try:
                    length = len(data)
                except TypeError:
                    # data is a scalar
                    length = 1
                    data = [data]
            self.values = data
            if length > 0:
                self.dtype = type(data[0])
                self.shape = (length,)

    def to_list(self):
        return self.values


#
# class AssociativeArray(GmxapiDataHandle):
#     """Associative data structure providing a key-value map.
#
#     Instances more closely resemble Python `collections.namedtuple` objects than
#     native Python `dict` objects. However, fields are typed, and objects must
#     be initialized with appropriately typed values or Futures for each field.
#     """
#     def __init__(self, map):
#         assert isinstance(map, dict)
#         self._data = {key: value for key, value in map.items()}
#         self._data_description = DataDescription(name='', dtype=AssociativeArray, shape=(1,))
#
#     @property
#     def data(self):
#         return self._data
#
#     @property
#     def data_description(self):
#         return self._data_description


# class DataDescription(object):
#     """A description of data that may or may not already exist.
#
#     Data instances have more definite features than can be inferred from the
#     type definition alone. Instances of DataDescription provide a way to describe
#     data the will be produced or consumed, including information that may exceed
#     what can be inferred from type alone. Contrast with GmxapiDataHandle, which
#     describes the minimal interface for all gmxapi data types.
#     """
#     def __init__(self, name: str, dtype: GmxapiDataHandle, shape=()):
#         self._name = name
#         assert issubclass(dtype, GmxapiDataHandle)
#         self._dtype = dtype
#         assert isinstance(shape, tuple)
#         self._ensemble_shape = shape
#
#     @classmethod
#     def create(cls, name: str, dtype=None, shape: tuple = (1,)):
#         # Here, "shape" refers to the ensemble topology of the source or sink.
#         if not isinstance(name, str):
#             raise exceptions.TypeError('Output descriptions are keyed by Python strings.')
#         if not isinstance(dtype, type):
#             if isinstance(dtype, DataDescription):
#                 return DataDescription(name, dtype=dtype.gmxapi_datatype, shape=dtype.ensemble_shape)
#             # Flavor might be inferred from a type hint return annotation.
#             assert dtype == False
#         else:
#             if issubclass(dtype, GmxapiDataHandle):
#                 return dtype.data_description
#             else:
#                 # Try to deduce a few types.
#                 if issubclass(dtype, Iterable):
#                     if issubclass(dtype, Mapping):
#                         # TODO: stronger type checking
#                         dtype = AssociativeArray
#                     elif issubclass(dtype, (str, bytes)):
#                         dtype = String
#                     else:
#                         # NDArray outputs are not automatically scattered to ensembles
#                         # TODO: stronger typing
#                         dtype = NDArray
#                 elif issubclass(dtype, bool):
#                     dtype = Boolean
#                 elif issubclass(dtype, float):
#                     dtype = Float64
#                 elif issubclass(dtype, int):
#                     dtype = Integer64
#                 else:
#                     raise exceptions.ValueError('Cannot infer gmxapi data type from {}.'.format(dtype))
#                 return DataDescription(name, shape=shape, dtype=dtype)
#
#     @property
#     def ensemble_shape(self):
#         return self._ensemble_shape
#
#     @property
#     def gmxapi_datatype(self):
#         return self._dtype

class ResultDescription:
    """Describe what will be returned when `result()` is called."""

    def __init__(self, dtype=None, width=1):
        assert isinstance(dtype, type)
        assert issubclass(dtype, (str, bool, int, float, dict, NDArray))
        assert isinstance(width, int)
        self._dtype = dtype
        self._width = width

    @property
    def dtype(self) -> type:
        """node output type"""
        return self._dtype

    @property
    def width(self) -> int:
        """ensemble width"""
        return self._width


class EnsembleDataSource(object):
    """A single source of data with ensemble data flow annotations."""
    def __init__(self):
        self.source = None
        self.width = 1
        self.dtype = None

    def node(self, member: int):
        return self.source[member]


class DataSourceCollection(collections.OrderedDict):
    """Store and describe input data handles for an operation.


    """

    def __init__(self, **kwargs):
        """Initialize from key/value pairs of named data sources.

        Data sources may be any of the basic gmxapi data types, gmxapi Futures
        of those types, or gmxapi ensemble data bundles of the above.

        Note that the checking and conditioning could be moved to one or more
        creation functions. In conjunction with an InputCollectionDescription,
        such creation functions could make decisions about automatically shaping
        the data flow topology or making conversions of data type or shape.
        """
        named_data = []
        for name, value in kwargs.items():
            if not isinstance(name, str):
                raise exceptions.TypeError('Data must be named with str type.')
            if not isinstance(value, (str, bool, int, float, dict, NDArray, EnsembleDataSource)):
                if isinstance(value, collections.abc.Iterable):
                    # Note: Here we assume that iterables are arrays first and ensemble data if necessary.
                    value = ndarray(value)
            named_data.append((name, value))
        super().__init__(named_data)


def ndarray(data=None):
    """Create an NDArray object from the provided iterable.

    Arguments:
        data: object supporting sequence, buffer, or Array Interface protocol
        shape: integer tuple specifying the size of each of one or more dimensions
        dtype: data type understood by gmxapi

    If `data` is provided, `shape` and `dtype` are optional. If `data` is not
    provided, both `shape` and `dtype` are required.

    If `data` is provided and shape is provided, `data` must be compatible with
    or convertible to `shape`. See Broadcast Rules in :doc:`datamodel` documentation.

    If `data` is provided and `dtype` is not provided, data type is inferred
    as the narrowest scalar type necessary to hold any element in `data`.
    `dtype`, whether inferred or explicit, must be compatible with all elements
    of `data`.

    The returned object implements the gmxapi N-dimensional Array Interface.

    ToDo: Does ndarray accept Futures in data and produce a Future? Or does it
     reject Futures in input? In other words, is ndarray guaranteed to produce a
     locally available object, and, if so, does it do so by rejecting non-local
     data, by implicitly resolving data locally, or with behavior controlled by
     additional keywords? Suggestion (MEI): uniform local-only behavior; separate
     function(s) for other use cases, a la numpy.asarray()
    """
    # assert isinstance(shape, tuple)
    if data is None:
        # if shape is None or dtype is None:
        #     raise exceptions.ValueError('If data is not provided, both shape and dtype are required.')
        array = NDArray()
        # array.dtype = dtype
        # array.shape = shape
        # array.values = [dtype()] * shape[0]
    else:
        if isinstance(data, NDArray):
            # if shape is not None:
            #     assert shape == data.shape
            # if dtype is not None:
            #     assert dtype == data.dtype
            return data
        # data is not None, but may still be an empty sequence.
        length = 0
        try:
            length = len(data)
        except TypeError:
            # data is a scalar
            length = 1
            data = [data]
        # if len(shape) > 0:
        #     if length > 1:
        #         assert length == shape[0]
        # else:
        #     if length == 0:
        #         shape = ()
        #     else:
        #         shape = (length,)
        # if len(shape) > 0:
        #     if dtype is not None:
        #         assert isinstance(dtype, type)
        #         assert isinstance(data[0], dtype)
        #     else:
        #         dtype = type(data[0])
        # for value in data:
        #     if dtype == NDArray:
        #         assert False
        #     assert dtype(value) == value
        array = NDArray(data)
        # array.dtype = dtype
        # array.shape = shape
        # array.values = list([dtype(value) for value in data])
    return array
