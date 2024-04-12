import time
import warnings
from typing import List


class timer():
    def __init__(self):
        self._startt = time.time()

    def start(self):
        self._startt = time.time()

    def stop(self):
        timens = time.time() - self._startt
        return timens


def shapes2str(shapes: List):
    str_list = [str(shape) for shape in shapes]
    result_str = '[' + ', '.join(str_list) + ']'
    return result_str


def tuple2str(t: tuple, splitch=','):
    s = ''
    for i, v in enumerate(t):
        s += str(v)
        if i != len(t) - 1:
            s += splitch
    return s


# modify from https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/registry.py # noqa: E501
class Registry():
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def __setitem__(self, name, obj):
        # assert (name not in self._obj_map), (f"An object named '{name}' was already registered "
        #                                      f"in '{self._name}' registry!")
        if name in self._obj_map:
            warnings.warn(
                f"An object named '{name}' was already registered "
                f"in '{self._name}' registry!")
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self.__setitem__(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self.__setitem__(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        # if ret is None:
        # raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __getitem__(self, item):
        if self._obj_map.__contains__(item) is False:
            raise KeyError(f"No object named '{item}' found in '{self._name}' registry!")
        return self._obj_map[item]

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()

class GlobalVars():
    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def __getitem__(self, item):
        if not self.__contains__(item):
            return None
        return self._obj_map[item]

    def __setitem__(self, key, value):
        self._obj_map[key] = value

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()
