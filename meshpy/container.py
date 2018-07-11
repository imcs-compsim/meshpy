# -*- coding: utf-8 -*-
"""
This module implements containers to manage boundary conditions and geometry
sets in one object.
"""

# Python modules.
from _collections import OrderedDict

# Meshpy modules.
from . import mpy, GeometrySet


class GeometryName(dict):
    """
    Group node geometry sets together. This is mainly used for export from mesh
    functions. The sets can be accessed by a unique name. There is no
    distinction between different types of geometry, every name can only be used
    once -> use meaningful names.
    """

    def __setitem__(self, key, value):
        """Set an geometry set in this container."""

        if not isinstance(key, str):
            raise TypeError('Expected string, got {}!'.format(type(key)))
        elif isinstance(value, GeometrySet):
            dict.__setitem__(self, key, value)
        else:
            raise ValueError('TODO')
            dict.__setitem__(self, key, GeometrySet(nodes=value))


class BoundaryConditionContainer(OrderedDict):
    """
    A class to group boundary conditions together. The key of the dicitonary
    are (bc_type, geometry_type).
    """
    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)

        for bc_key in mpy.boundary_condition:
            for geometry_key in mpy.geometry:
                self[(bc_key, geometry_key)] = []


class GeometrySetContainer(OrderedDict):
    """
    A class to group geometry sets together with the key being the geometry
    type.
    """
    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)

        for geometry_key in mpy.geometry:
            self[geometry_key] = []




# 
# 
# 
# class Container(OrderedDict):
#     """
#     A base class for a container that will store node sets and   boundary conditions.
#     """
#     
#     def __init__(self, fields, empty_type):
#         """ Create a dictionary with the keys. """
#         
#         self._fields = fields
#         
#         # set empty dictionary
#         for field in self._fields:
#             self[field[0]] = empty_type()
#     
#     
#     def __getattr__(self, name):
#         """ Check if attribute is in self._fields. """
#         for field in OrderedDict.__getattribute__(self, '_fields'):
#             if name in field:
#                 return self[field[0]]
#         raise AttributeError('Attribute {} does not exist!'.format(name))
#     
#     
#     def __setattr__(self, *args, **kwargs):
#         return OrderedDict.__setattr__(self, *args, **kwargs)
#     
#     
#     def __iter__(self):
#         """ Return the items in fields. """
#         for key in self.keys():
#             yield self[key]    








    
    #def __attr
#     
#     def empty_item(self):
#         """ What will be in the default items. """
#         return []
# 
#     def _get_key(self, key):
#         """ Return the key for the dictionary. Look in self.aliases. """
#         for line in self.aliases:
#             if key == line[0]:
#                 return line[0]
#             elif key in line[1]:
#                 return line[0]
#         print('Error, key {} not found!'.format(key))
#     
#     
#     def merge_containers(self, other_container):
#         """ Merge the contents of this set with a other SetContainer. """
#         if type(other_container) == type(self):
#             for key in other_container.keys():
#                 self[key].extend(other_container[key])
#         else:
#             print('Error, expected type {}, got {}!'.format(type(self), type(other_container)))
#     
#     def set_global(self):
#         """ Set the global values in each set element. """
#         for key in self.keys():
#             for i, item in enumerate(self[key]):
#                 item.n_global = i + 1
#             
#     def __setitem__(self, key, value):
#         """ Set items of the dictionary. """
#         dict_key = self._get_key(key)
#         OrderedDict.__setitem__(self, dict_key, value)
#         
#     
#     def __getitem__(self, key):
#         """ Gets items of the dictionary. """
#         dict_key = self._get_key(key)
#         return OrderedDict.__getitem__(self, dict_key)
#     
#     
#     def append_item(self, key, data):
#         """ Add set(s) to object. Set the type of set in the item. """
#         data_type = self._get_key(key)
#         if type(data) == list:
#             self[key].extend(data)
#             for item in data:
#                 item.item_type = data_type
#         else:
#             self[key].append(data)
#             data.item_type = data_type












# 
# 
# 
# class ContainerGeom(Container):
#     def __init__(self):
#         Container.__init__(self, mpy.geo, list)
# class ContainerBC(Container):
#     def __init__(self):
#         Container.__init__(self, mpy.bc, ContainerGeom)
# 
# 










