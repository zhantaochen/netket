import os


def bool_env(varname: str, default: bool) -> bool:
    """Read an environment variable and interpret it as a boolean.
    True: 'y', 'yes', 't', 'true', 'on', and '1';
    False: 'n', 'no', 'f', 'false', 'off', and '0'.

    Args:
        varname: the name of the variable
        default: the default boolean value
    """
    val = os.getenv(varname, str(default))
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r for environment %r" % (val, varname))


def int_env(varname: str, default: int) -> int:
    """Read an environment variable and interpret it as an integer."""
    return int(os.getenv(varname, default))


def get_env(varname: str, type, default: int):
    if type is int:
        return int_env(varname, default)
    elif type is bool:
        return bool_env(varname, default)
    else:
        raise TypeError(f"Unknown type {type}")


class ReadOnlyDict:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

        # def __contains__(self, key)
        key in self._data


class Config:
    def __init__(self):
        self._values = {}
        self._types = {}
        self._editable_at_runtime = {}

        self._readonly = ReadOnlyDict(self._values)

    def define(self, name, type, default, *, help, runtime=False):
        """
        Defines a new flag
        """
        if name in self._values:
            raise KeyError(f"Flag {name} already defined.")

        self._types[name] = type
        self._editable_at_runtime[name] = runtime
        self._values[name] = get_env(name, type, default)

    @property
    def FLAGS(self):
        """
        The flags of this instance of netket
        """
        return self._readonly

    def update(self, name, value):
        """
        Updates a variable in netket

        Args:
            name: the name of the variable
            value: the new value
        """
        self._values[name] = self._types[name](value)


config = Config()
FLAGS = config.FLAGS

config.define(
    "NETKET_DEBUG",
    bool,
    default=False,
    help="Enable debug logging in many netket functions.",
    runtime=True,
)

config.define(
    "NETKET_EXPERIMENTAL",
    bool,
    default=False,
    help="Enable experimental features.",
    runtime=False,
)
