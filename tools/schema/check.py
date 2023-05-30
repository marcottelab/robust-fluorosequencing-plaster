"""
Lots more guidelines here. When to replace assert etc.
Don't check your own arguemn
Strict guidelines.

"""

import inspect
import os
import re
from functools import wraps
from typing import get_type_hints

import numpy as np
import pandas as pd


def _context(depth=2):
    frame = inspect.currentframe()
    try:
        f = frame
        while depth > 0:
            f = f.f_back
            depth -= 1
        context = inspect.getframeinfo(f)
        callers = "".join([line.strip() for line in context.code_context])
        m = re.search(r"\((.+?)\)$", callers)
        if m:
            return (
                [i.strip() for i in m.group(1).split(",")][0],
                context.filename,
                context.lineno,
            )

        return "", context.filename, context.lineno

    finally:
        del frame


class CheckError(Exception):
    def __init__(self, expected_type, was_type, depth=3, message=None):
        var_name, source, lineno = _context(depth=depth)

        expected_type_name = expected_type
        if hasattr(expected_type, "__name__"):
            expected_type_name = expected_type.__name__

        was_type_name = was_type
        if hasattr(was_type, "__name__"):
            was_type_name = was_type.__name__

        if message is None:
            message = f"{var_name} was expected to be type '{expected_type_name}' but was type '{was_type_name}'"
        else:
            message = message.format(**locals())

        super().__init__()
        self.var_name = var_name
        self.was_type = was_type
        self.expected_type = expected_type
        self.source = source
        self.lineno = lineno
        self.message = message

    def __deepcopy__(self, memodict={}):
        return CheckError(self.expected_type, self.was_type, message=self.message)

    def __repr__(self):
        return f"Check failed in {os.path.basename(self.source)}:{self.lineno}, {self.message}\n"

    def __str__(self):
        return self.__repr__()


class CheckAffirmError(TypeError):
    def __init__(self, depth=3, message=None):
        _, source, lineno = _context(depth=depth)

        if message is not None:
            message = message.format(**locals())

        super().__init__()
        self.source = source
        self.lineno = lineno
        self.message = message

    def __repr__(self):
        return f"Affirmation failed in {os.path.basename(self.source)}:{self.lineno}, {self.message}\n"

    def __str__(self):
        return self.__repr__()


def _print(msg):
    """mock-point"""
    print(msg)


def affirm(condition, message=None, exp=CheckAffirmError):
    """
    Like assert, but raises messages that are easy to test.
    If cond is false then raise exp. Jam message into the exception if not None.

    Use this as opposed to assert when you want to make a test that checks
    that this condition is enforced. Using the message argument allows the
    test to check that a specific affirmation was raised not just *any* assert.
    """
    if not condition:
        if not isinstance(exp, Exception):
            exp = exp()
        if message is not None:
            exp.message = message
        raise exp


def t(instance, expected_type, depth=3, **kwargs):
    """
    Check that instance is of expected_type and produce a useful trace and raise if not.

    Usage:
    from plaster.tools.schema import check

    def some_func(a):
        check.t(a, int)
        check.t(a, (int, float, None))  # May use a tuple of acceptable types
    """
    # CHANGE None to NoneType
    if isinstance(expected_type, tuple):
        expected_type = tuple([t or type(None) for t in expected_type])
    elif expected_type is None:
        expected_type = type(None)

    if not isinstance(instance, expected_type):
        raise CheckError(expected_type, type(instance), depth, **kwargs)


def list_t(instance, expected_type):
    """
    Checks that instance is a list and that all elements are of expected_type
    """
    t(instance, list, depth=4)
    for i in instance:
        t(i, expected_type, depth=4)


def list_or_tuple_t(instance, expected_type, expected_len=None):
    """
    Checks that instance is a list or tuple and that all elements are of expected_type
    """
    t(instance, (list, tuple), depth=4)
    if expected_len is not None and expected_len != len(instance):
        raise CheckAffirmError(
            message=f"Expected list or tuple of length {expected_len}, got {len(instance)}"
        )

    for i in instance:
        t(i, expected_type, depth=4)


def df_t(instance, df_schema, allow_extra_columns=False):
    """
    Check the columns of instance match the df_schema
    Like dict(col_a=dtype("float64))
    """
    t(instance, pd.DataFrame, depth=4)
    for col in instance.columns:
        if col in df_schema:
            if instance[col].dtype != df_schema[col]:
                message = (
                    "DataFrame '{var_name}' was expected to have column '"
                    + str(col)
                    + "' of type '"
                    + str(df_schema[col])
                    + "' but had type '"
                    + str(instance[col].dtype)
                    + "'"
                )
                raise CheckError(
                    df_schema[col], instance[col].dtype, depth=3, message=message
                )

        elif not allow_extra_columns:
            raise CheckError(
                None, None, depth=3, message=f"Extra column '{col}' found in dataframe."
            )


def array_t(
    instance, shape=None, dtype=None, ndim=None, is_square=None, c_contiguous=None
):
    """
    If no shape is passed in, it simply prints the dimensions which
    is particularly handy when working in a notebook context.
    """
    t(instance, np.ndarray, depth=4)

    if ndim is not None and instance.ndim != ndim:
        message = (
            "ndarray [shape="
            + str(instance.shape)
            + "] was expected to be ndim '"
            + str(ndim)
            + "' but was '"
            + str(instance.ndim)
            + "'"
        )
        raise CheckError(dtype, instance.dtype, depth=3, message=message)

    if dtype is not None and instance.dtype != dtype:
        message = (
            "ndarray [shape="
            + str(instance.shape)
            + "] was expected to be dtype '"
            + str(dtype.__name__)
            + "' but was dtype '"
            + str(instance.dtype)
            + "'"
        )
        raise CheckError(dtype, instance.dtype, depth=3, message=message)

    if shape is not None:
        if len(shape) != len(instance.shape):
            message = (
                "ndarray was expected to have "
                f"{len(shape)} dimensions had {len(instance.shape)}"
            )
            raise CheckError(dtype, instance.dtype, depth=3, message=message)

        for i, (expected, actual) in enumerate(zip(shape, instance.shape)):
            if expected is not None:
                if expected != actual:
                    message = (
                        "ndarray dimension "
                        f"{i} was expected to be {expected} but was "
                        f"{actual}"
                    )
                    raise CheckError(dtype, instance.dtype, depth=3, message=message)

    if is_square is not None and (
        instance.shape[0] != instance.shape[1] or instance.ndim != 2
    ):
        message = "ndarray was expected to be square but was " + str(instance.shape)
        raise CheckError(dtype, instance.dtype, depth=3, message=message)

    if c_contiguous is not None and not instance.flags["C_CONTIGUOUS"]:
        message = "ndarray was expected to be c_contiguous but was not"
        raise CheckError(dtype, instance.dtype, depth=3, message=message)

    if (
        shape is None
        and dtype is None
        and ndim is None
        and is_square is None
        and c_contiguous is None
    ):
        # Print information shape without asserting
        context = _context(depth=2)
        _print(f"'{context[0]}': shape={instance.shape} dtype={instance.dtype}")


def args(fn):
    """
    Use to enforce argument type hints

    Example:
        from plaster.tools.schema import check

        @check.args
        def some_func(n_row:int, help=""):
            something()

        some_func("a")  # raises a CheckError
    """

    @wraps(fn)
    def wrapper(*_args, **_kwargs):
        hints = get_type_hints(fn)
        signature = inspect.signature(fn)
        actual_args = signature.bind(*_args, **_kwargs).arguments
        for name, parameter in signature.parameters.items():
            if name in hints:
                arg = actual_args.get(name)
                if not isinstance(arg, hints.get(name)):
                    message = (
                        "When calling '"
                        + fn.__name__
                        + "(...)' argument '"
                        + name
                        + "' was expected to be type '"
                        + hints.get(name).__name__
                        + "' but was called with type '"
                        + type(arg).__name__
                        + "'"
                    )
                    raise CheckError(hints.get(name), type(arg), message=message)

        return fn(*_args, **_kwargs)

    return wrapper
