import dataclasses
import hashlib
import inspect
import json
import logging
import math
import os
import pickle
import random
import re
import readline
import resource
import string
import struct
import sys
import termios
import textwrap
import threading
import time
import tty
import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
import yaml
from munch import Munch
from plumbum import local

from plaster.tools.schema import check
from plaster.tools.zlog.zlog import terminal_size

log = logging.getLogger(__name__)


def get_cursor_pos():
    """https://stackoverflow.com/questions/46651602/determine-the-terminal-cursor-position-with-an-ansi-sequence-in-python-3"""
    buf = ""
    stdin = sys.stdin.fileno()
    tattr = termios.tcgetattr(stdin)

    try:
        tty.setcbreak(stdin, termios.TCSANOW)
        sys.stdout.write("\x1b[6n")
        sys.stdout.flush()

        while True:
            buf += sys.stdin.read(1)
            if buf[-1] == "R":
                break

    finally:
        termios.tcsetattr(stdin, termios.TCSANOW, tattr)

    # reading the actual values, but what if a keystroke appears while reading
    # from stdin? As dirty work around, getpos() returns if this fails: None
    try:
        matches = re.match(r"^\x1b\[(\d*);(\d*)R", buf)
        groups = matches.groups()
    except AttributeError:
        return None

    return (int(groups[0]), int(groups[1]))


def quiet_rm_file(path):
    """Remove and ignore errors"""
    try:
        os.remove(path)
    except Exception:
        pass


def smart_wrap(text, width=80, assert_if_exceeds_width=False):
    """
    Useful when doing indented multi-line like:
        long_str = "Thi is a really long string"
        smart_wrap(f'''
            This first line sets the indent...
            And any other lines have their indent maintained
                Example, this {long_str} is really long it will
                wrap at this particular indent.
            And now back to the un-indented
        ''')

        The 60 is chosen to be reasonable for many applications
        but in a jupyter notebook the terminal returns too large a width.

    Arguments:
        text: See above
        width: wrap width or "auto" or None
            When width is None it performs the unindent but not the wrap
    """

    if width == "auto":
        width = terminal_size()[0]

    res_lines = []
    lines = text.split("\n")
    first_indent = re.search(r"\S", lines[1]).start()
    for line_i, line in enumerate(lines):
        indent = 0
        search = re.search(r"\S", line)
        if search is not None:
            indent = search.start()
        if width is None:
            sublines = [line[indent:]]
        else:
            _line = line[indent:]
            if assert_if_exceeds_width:
                if len(_line) > width:
                    raise ValueError(
                        f"exceeds width line {line_i} (was {len(_line)}) (starts with '{_line[0:20]}')"
                    )
            sublines = textwrap.wrap(_line, width)
        if len(sublines) == 0:
            res_lines += [""]
        for subline in sublines:
            res_lines += [" " * (indent - first_indent) + subline]

    return "\n".join(res_lines)


def smart_print(text, width=80):
    print(smart_wrap(text, width))


def listi(list_, elem, default=None):
    """
    Return the elem component in a list of lists, or list of tuples, or list of dicts.
    If default is non-None then if the key is missing return that.

    Examples:
        l = [("A", "B"), ("C", "D")]
        listi(l, 1) == ["B", "D"]

        l = [{"A":1, "B":2}, {"A":3, "B", 4}, {"Q": 5}]
        listi(l, "B", default=0) == [2, 4, 0]
    """
    ret = []
    for i in list_:
        if isinstance(i, dict):
            if elem in i:
                ret += [i[elem]]
            elif default is not None:
                ret += [default]
            else:
                raise KeyError("Missing elem in list of dicts")
        else:
            if 0 <= elem < len(i):
                ret += [i[elem]]
            elif default is not None:
                ret += [default]
            else:
                raise KeyError("Missing elem in list of lists")

    return ret


def ensure_list(i):
    """If i is a singleton, convert it to a list of length 1"""

    # Not using isinstance here because an object that inherits from a list
    # is not considered a list here. That is, I only want to compare on the final-type.
    if type(i) is list:
        return i
    return [i]


def pad_list(list_, len_, pad=None):
    n_pad_needed = len_ - len(list_)
    if n_pad_needed > 0:
        return list_ + [pad] * n_pad_needed
    return list_


def filt_first(list_, func):
    """
    Like filter but reverse arguments and it expects there to be only one result,
    None on not found.

    Example:
        my_list = [(1, 2), (3, 4)]
        ret = filt_first(my_list, lambda x: x[1] == 4)
        assert ret == (3, 4)
    """
    try:
        return next(i for i in list_ if func(i))
    except StopIteration:
        return None


def filt_last(list_, func):
    """Like above but finds last"""
    matching = [i for i in list_ if func(i)]
    if len(matching) > 0:
        return matching[-1]
    return None


def filt_all(list_, func):
    """Like filter but reverse arguments and returns list"""
    return [i for i in list_ if func(i)]


def filt_first_arg(list_, func):
    """Like filt_first but return index (arg) instead of value"""
    for i, x in enumerate(list_):
        if func(x):
            return i
    return None


def filt_last_arg(list_, func):
    """Like filt_last but return index (arg) instead of value. Inefficiently traverses whole list"""
    last_arg = None
    for i, x in enumerate(list_):
        if func(x):
            last_arg = i
    return last_arg


def filt_all_arg(list_, func):
    """Like filt_all but return indices (args) instead of value"""
    ret = []
    for i, x in enumerate(list_):
        if func(x):
            ret += [i]
    return ret


def bound(val, _min, _max):
    return _min if val < _min else _max if val > _max else val


def safe_list_get(l, offset, default=None):
    try:
        return l[offset]
    except (IndexError, TypeError):
        return default


def safe_len(l, default=0):
    try:
        return len(l)
    except Exception:
        return default


def safe_del(dict_, key):
    try:
        del dict_[key]
    except Exception:
        pass


def flatten(l, depth=None):
    def _flatten(l, depth):
        for el in l:
            if isinstance(el, (list, tuple)) and (depth is None or depth > 0):
                yield from _flatten(el, depth - 1 if depth is not None else None)
            else:
                yield el

    return list(_flatten(l, depth))


def set_defaults(dic, **kwargs):
    """
    If keys are not already in the dic then add it with the val.
    This adds the keys to the existing dic, (does not copy!)

    Example:

    def something(**kwargs):
        # If kwargs doesn't already have "abc" and "def" then set them.
        kwargs = set_defaults(kwargs, abc=123, def=456)
        do_something(**kwargs)
    """
    for key, val in kwargs.items():
        if key not in dic:
            dic[key] = val
    return dic


def sample(parameter_funcs, trial_func, n_trials):
    """
    Sample some function with a set of parameters.
    Example usage:

        def trial(a, b, c):
            return something_interesting(a,b,c)

        results = stats.sample(
            (
                # peak_std
                lambda size: np.random.uniform(1.0, 2.0, size=size),

                # hat_rad
                lambda size: np.random.randint(1, 3, size=size),

                # brim_rad
                lambda size: np.random.randint(1, 3, size=size),
            ),
            trial,
            n_trials=5,
        )

        Where results is a tuple. The [0] value of the tuple is the result and
        the remaining parts are the parameters that were sampled

    """
    samples_by_parameter = []
    for p_func in parameter_funcs:
        samples_by_parameter += [p_func(size=n_trials)]

    values = list(map(trial_func, *samples_by_parameter))
    return list(zip(values, *samples_by_parameter))


class Timer:
    def __init__(self, msg=None, show_start=False):
        self.start = time.time()
        self.stop = None
        self.msg = msg
        self.show_start = show_start

    def __enter__(self):
        if self.msg is not None and self.show_start:
            print(f"Start: '{self.msg}'", flush=True)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = time.time()
        if self.msg is not None:
            print(
                f"Done: '{self.msg}' in {self.stop - self.start:.1f} secs.", flush=True
            )

    def elapsed(self):
        if self.stop is None:
            return time.time() - self.start
        return self.stop - self.start


def indexed_pickler_dump(obj, path, prop_list=None):
    """
    This pickles the properties of an object into an indexed pickle file
    so that they can be selectively loaded later.

    Arguments:
        obj: an instance of a class or dict to save properties of
        path: is the file to save
        prop_list: the set of properties to save, or None if you want
            to save all properties of the class
    """

    if isinstance(obj, dict):
        pickled_parts = {
            prop: pickle.dumps(obj[prop], protocol=4)
            for prop in obj.keys()
            if prop_list is None or prop in prop_list
        }
    else:
        pickled_parts = {
            prop: pickle.dumps(getattr(obj, prop), protocol=4)
            for prop in obj.__dict__.keys()
            if prop_list is None or prop in prop_list
        }

    # Serialize the parts one after the other and make an index
    serialized = bytes()
    index = {}
    for prop, pick in sorted(pickled_parts.items()):
        index[prop] = (len(serialized), len(pick))
        serialized += pick

    pickled_index = pickle.dumps(index, protocol=4)
    with open(path, "wb") as f:
        f.write(struct.pack("Q", len(pickled_index)))
        f.write(pickled_index)
        f.write(serialized)


def indexed_pickler_load(
    path, prop_list=None, into_instance=None, skip_missing_props=False
):
    """
    Load the selected properties from the indexed pickle file.
    The properties wil be set on into_instance if specified

    Arguments:
        path: file to load from
        prop_list: If a list, a dict is returned, if None all keys, else one value
        into_instance: If non None, insert the prop, val as an attribute into this
        skip_missing_props: if True, ignore the missing properties
    """
    is_singleton = False
    pickled_props = {}
    with open(path, "rb") as f:
        index_len = struct.unpack("Q", f.read(8))[0]
        index = pickle.loads(f.read(index_len))
        start = f.tell()

        _prop_list = prop_list
        if _prop_list is None or _prop_list == [None]:
            _prop_list = list(index.keys())

        if not isinstance(_prop_list, (list, tuple)):
            is_singleton = True
            _prop_list = [prop_list]

        for prop in sorted(_prop_list):
            offset = index.get(prop)
            if offset is None and skip_missing_props:
                continue
            f.seek(offset[0] + start)
            pickled_prop = f.read(index[prop][1])
            pickled_props[prop] = pickle.loads(pickled_prop)
            if into_instance is not None:
                setattr(into_instance, prop, pickled_props[prop])

        if is_singleton:
            # De-reference singleton
            return pickled_props[prop_list]

        return pickled_props


def indexed_pickler_load_keys(path):
    """
    Return the list of keys in the indexed pickle
    """
    with open(path, "rb") as f:
        index_len = struct.unpack("Q", f.read(8))[0]
        return pickle.loads(f.read(index_len)).keys()


def pickle_load(path):
    with open(local.path(path), "rb") as f:
        return pickle.load(f)


def pickle_load_munch(path):
    m = pickle_load(path)
    if isinstance(m, Munch):
        return m
    else:
        return Munch.fromDict(m)


def pickle_write(path, **kwargs):
    with open(local.path(path), "wb") as f:
        pickle.dump(Munch(**kwargs), f, protocol=4)


def pickle_save(path, obj):
    with open(local.path(path), "wb") as f:
        pickle.dump(obj, f, protocol=4)


def json_load(path):
    with open(local.path(path), "rb") as f:
        return json.loads(f.read())


def json_load_munch(path):
    return Munch.fromDict(json_load(path))


class JSONDataClassEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def json_write(path, **kwargs):
    with open(local.path(path), "w") as f:
        f.write(json.dumps(Munch(**kwargs), indent=4, cls=JSONDataClassEncoder))


def json_save(path, dict_or_list):
    with open(local.path(path), "w") as f:
        f.write(json.dumps(dict_or_list, indent=4, cls=JSONDataClassEncoder))


def json_print(dict_or_list):
    print(
        json.dumps(
            dict_or_list,
            indent=4,
            sort_keys=True,
            default=str,
            cls=JSONDataClassEncoder,
        )
    )


class SubclassDumper(yaml.Dumper):
    """Force subclasses of strings and dicts to dump natively"""

    ignore_aliases = lambda *args: True

    def represent_data(self, data):
        if dataclasses.is_dataclass(data):
            return self.represent_dict(dataclasses.asdict(data))
        if isinstance(data, dict):
            return self.represent_dict(data)
        if isinstance(data, str):
            return self.represent_str(data)
        if isinstance(data, tuple):
            return self.represent_list(data)
        return super().represent_data(data)


def yaml_load(path):
    with open(local.path(path), "r") as f:
        return yaml.safe_load(f)


def yaml_load_munch(path):
    return Munch.fromDict(yaml_load(path)) or Munch.fromDict({})


def yaml_save(path, dict_or_list):
    with open(local.path(path), "w", encoding="utf8") as f:
        yaml.dump(
            dict_or_list, f, Dumper=SubclassDumper, width=60, default_flow_style=False
        )


def yaml_write(path, **kwargs):
    yaml_save(path, Munch(**kwargs))


def yaml_print(dict_or_list):
    print(
        yaml.dump(dict_or_list, default_flow_style=False, allow_unicode=True, indent=4)
    )


def block_all_key_vals(dict_or_list, parent_name=""):
    found = []
    if isinstance(dict_or_list, list):
        if parent_name != "":
            parent_name += "."
        for i, o in enumerate(dict_or_list):
            node_name = f"{parent_name}{i}"
            found += block_all_key_vals(o, node_name)
    elif isinstance(dict_or_list, dict):
        if parent_name != "":
            parent_name += "."
        for k, v in dict_or_list.items():
            node_name = f"{parent_name}{k}"
            found += block_all_key_vals(v, node_name)
    else:
        return [(parent_name, dict_or_list)]

    return found


def block_all_keys(dict_or_list, parent_name=""):
    found = []
    if parent_name != "":
        parent_name += "."
    if isinstance(dict_or_list, list):
        for i, o in enumerate(dict_or_list):
            node_name = f"{parent_name}{i}"
            found += [node_name]
            found += block_all_keys(o, node_name)
    elif isinstance(dict_or_list, dict):
        for k, v in dict_or_list.items():
            node_name = f"{parent_name}{k}"
            found += [node_name]
            found += block_all_keys(v, node_name)
    else:
        return []

    return found


def block_search(dict_or_list, block_name, default=None):
    """Find block_name in dict_or_list where block_name in the form 'keya.keyb.keyc'"""
    current = dict_or_list
    if block_name is None or block_name == "" or block_name == ".":
        return current
    for part in block_name.split("."):
        if isinstance(current, list):
            try:
                idx = int(part)
            except ValueError:
                return default
            if 0 <= idx < len(current):
                current = current[idx]
            else:
                return default
        elif isinstance(current, dict):
            current = current.get(part)
            if current is None:
                return default
    return current


def block_update(dict_or_list, block_name, value):
    """
    Add the elements to the structure as needed to set the value.

    Example:
        d = dict(a=dict(b=2), c=3)
        block_update(d, "a.b", 4)
        # Will change b from 2 to 4
    """

    parent = None
    parts = block_name.split(".")

    # Scan through the existing struct and try to resolve as much of the
    # parts as possible
    # Stop when either:
    #     * all parts are resolved. (total_resolution = True)
    #     * there's no matching element in the struct for the part (total_resolution = False)

    fully_resolved = False
    current = dict_or_list
    for i, part in enumerate(parts):
        if isinstance(current, list):
            # The current element is a list, does the part lookup into the list?
            try:
                idx = int(part)
            except ValueError:
                # Asking for a non-list element in a list
                break

            parent = current
            if 0 <= idx < len(current):
                # part found, continue to descend
                current = current[idx]
            else:
                # part not found
                break

        elif isinstance(current, dict):
            # The current element is a dict, does the part lookup into the dict?
            parent = current
            found = current.get(part)
            if found:
                # part found in current node, continue to descend
                current = found
            else:
                # part not found
                break

        else:
            # The node is a scalar, so we're done.
            # But we have to go back one in the parts because we may need to create
            # the container
            i -= 1
            break
    else:
        # Made it through the whole loop meaning that all parts were resolved
        # Therefore all we have to do is write into the parent the value to be set
        fully_resolved = True

    if not fully_resolved:
        # Continue through the parts creating new nodes, starting at i
        # Stop one early because we want to set the last node sing the setter below
        for i in range(i, len(parts) - 1):
            part = parts[i]

            idx = None
            try:
                idx = int(part)
            except ValueError:
                pass

            if idx is not None:
                parent[part] = []
            else:
                parent[part] = {}

            parent = parent[part]

    # All the parents were either already present or have been created
    if isinstance(parent, list):
        idx = int(parts[-1])
    else:
        idx = parts[-1]
    parent[idx] = value


def ipython_info():
    ip = False
    if "ipykernel" in sys.modules:
        ip = "notebook"
    elif "IPython" in sys.modules:
        ip = "terminal"
    return ip


def strip_underscore_keys(struct, munchify=False):
    if isinstance(struct, dict):
        block = {
            k: strip_underscore_keys(v, munchify=munchify)
            for k, v in sorted(struct.items())
            if not k.startswith("_")
        }
        if munchify:
            block = Munch.fromDict(block)
        return block
    elif isinstance(struct, list):
        return [strip_underscore_keys(v, munchify=munchify) for v in struct]
    else:
        return struct


def max_rss():
    # https://stackoverflow.com/questions/12050913/whats-the-unit-of-ru-maxrss-on-linux
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024
    return rss


def duplicates(iter):
    return list(set([x for x in iter if iter.count(x) > 1]))


def _escape_quotes(text, quote_character):
    return text.replace(quote_character, "\\" + quote_character)


def escape_single_quotes(text):
    return _escape_quotes(text, "'")


def escape_double_quotes(text):
    return _escape_quotes(text, '"')


def escape_spaces(text):
    return text.replace(" ", "\\ ")


def interactive_input_line_with_prefill(prompt, prefill=""):
    # Based on https://stackoverflow.com/questions/2533120/show-default-value-for-editing-on-python-input-possible/2533134
    readline.set_startup_hook(lambda: readline.insert_text(prefill))
    try:
        return input(prompt)
    finally:
        readline.set_startup_hook()


def interactive_confirm():
    return input().lower().startswith("y")


def random_str(length=6):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


def get_ecr_login_string():
    """
    Be sure to run the return. Example:
        from plaster.tools.utils.utils import get_ecr_login_string
        local["bash"]["-c", get_ecr_login_string()] & FG
    """
    return "$(aws ecr get-login --no-include-email --region=us-east-1) 2> /dev/null"


def elapsed_time_in_minutes_seconds(delta):
    return f"{int(delta // 60):d}:{int(delta) % 60:02d}"


def repl_vec_over_cols(vec, n_cols):
    """
    Given a vector of len n, replicated it into columns n_cols times.
    Example:
        repl_vec_over_cols(np.array([1, 2, 3]), 2) ==
        [
            [1, 1],
            [2, 2],
            [3, 3],
        ]
    """
    if vec.ndim == 1:
        vec = np.expand_dims(vec, 1)
        return np.tile(vec, (1, n_cols))
    elif vec.ndim == 2:
        if vec.shape[0] == 1:
            return np.tile(vec.T, (1, n_cols))
        elif vec.shape[1] == 1:
            return np.tile(vec, (1, n_cols))
        else:
            raise TypeError(
                "repl_vec_over_cols only works 2 dimensional vectors if one dimension has a shape of 1"
            )
    else:
        raise TypeError("repl_vec_over_cols only works on 1 or 2 dimensional vectors")


def load(file_path, return_on_non_existing=None):
    """
    Load a file. If return_on_non_existing is not None then
    if the file doesn't exist it returns that value otherwise
    it will exception as typical of open()
    """
    if return_on_non_existing is not None:
        if not local.path(file_path).exists():
            return return_on_non_existing

    with open(file_path) as f:
        return f.read()


def save(file_path, value):
    with open(file_path, "w") as f:
        return f.write(value)


def non_none(*args, raise_if_all_none=None):
    """
    Return the first arg that is not none; optionally raise specified exception if all none
    """
    for a in args:
        if a is not None:
            return a
    if raise_if_all_none is not None:
        raise raise_if_all_none
    return None


def ren_key(d, orig_key, new_key):
    d[new_key] = d.pop(orig_key)


def get_root_key(dict_):
    keys = list(dict_.keys())
    assert len(keys) == 1
    return keys[0]


def grid_to_csv(grid, float_fmt="6.3f"):
    """
    Usage:
        grid = np.empty((500, 300), dtype=object)
        grid[0:10, 0:5] = some_block_of_interest
        print(grid_to_csv(grid), file=open("test.csv", "w"))
    """
    assert grid.ndim == 2
    n_rows = (
        np_arg_last_where(np.array([any(grid_line != None) for grid_line in grid])) + 1
    )
    n_cols = (
        np.array(
            [np_arg_last_where(grid_line != None) for grid_line in grid], dtype=float
        )
        + 1
    )
    n_cols = int(np.nanmax(n_cols))
    lines = []
    for row in grid[0:n_rows]:
        fields = []
        for col in row[0:n_cols]:
            if isinstance(col, float):
                fields += [f"{col:{float_fmt}}"]
            elif isinstance(col, int):
                fields += [f"{col}"]
            elif isinstance(col, str):
                fields += [col]
            else:
                fields += [""]
        lines += [",".join(fields)]
    return "\n".join(lines)


def myself():
    """Get name of the current function"""
    return inspect.stack()[1][3]


def fourier_matrix(n):
    """
    Generate a small Fourier Matrix for a n-element vector.
    This would not a good idea for large transforms but it is simple and fast
    for small n such as he number of cycles for flurosequencing experiments.
    """
    ft_matrix = np.zeros((n, n))
    for i_data in range(n):
        ft_matrix[0, i_data] = 1 / math.sqrt(n)

    for kharm in range(int(math.ceil(n / 2) - 1)):
        for i_data in range(n):
            ft_matrix[2 * kharm + 1, i_data] = (
                -2.0
                * math.sin(2.0 * math.pi * (kharm + 1) * (i_data + 0.5) / n)
                / math.sqrt(2.0 * n)
            )
            ft_matrix[2 * kharm + 2, i_data] = (
                2.0
                * math.cos(2.0 * math.pi * (kharm + 1) * (i_data + 0.5) / n)
                / math.sqrt(2.0 * n)
            )

    if n % 2 == 0:
        for i_data in range(n):
            ft_matrix[n - 1, i_data] = math.sin(
                -2.0 * math.pi * n / 2 * (i_data + 0.5) / n
            ) / math.sqrt(n)

    return ft_matrix


def fourier_transform(mat):
    assert mat.ndim == 2
    ft_matrix = fourier_matrix(mat.shape[1])
    return (ft_matrix @ mat.T).T


def is_power_of_2(n: int):
    return n != 0 and ((n & (n - 1)) == 0)


def next_power_of_2(n: int):
    assert n >= 0 and int(n) == n
    if is_power_of_2(n):
        return 2 ** (math.log2(n) + 1)
    return 1 if n == 0 else 2 ** math.ceil(math.log2(n))


def prior_power_of_2(n: int):
    assert n > 1 and int(n) == n
    if is_power_of_2(n):
        return 2 ** (math.log2(n) - 1)
    return 2 ** math.floor(math.log2(n))


def normalize_to_square(n: int):
    """Normalize number to the nearest power of 2."""

    if is_power_of_2(n):
        return n

    next_power = next_power_of_2(n)
    prev_power = prior_power_of_2(n)

    return next_power if n >= prev_power * 1.5 else prev_power


symbol_pat = re.compile(r"^[a-z_][a-z0-9_]+$")


def is_symbol(str_):
    return symbol_pat.match(str_)


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def mat_flatter(mat):
    """
    Flatten to two dimensions.
    Eg: From (10, 20, 30) to (10, 60)
    """
    check.t(mat, np.ndarray)
    assert mat.ndim >= 2
    return mat.reshape(mat.shape[0], np.prod(mat.shape[1:]))


def mat_lessflat(mat, dim1=None, dim2=None):
    """
    To unflatten you must know either dim1 or dim2

    Example, suppose mat is (2, 6)

        m = mat_lessflat(mat, dim2=3)
        assert m.shape == (2, 2, 3)
    """
    check.array_t(mat, ndim=2)
    check.affirm(dim1 is not None or dim2 is not None)
    if dim1 is None:
        dim1 = mat.shape[1] // dim2
    if dim2 is None:
        dim2 = mat.shape[1] // dim1
    return mat.reshape(mat.shape[0], dim1, dim2)


file_lunch_lock = threading.Lock()


class Store(Munch):
    """
    This is a handy little class when manipulating results
    in a Notebook that you want to cache.

    cell1:
        # Load the store if it already exists
        store = Store()
        store = Store("filename.yaml")  # If

    cell2:
        # if you don't want to run this again, you can just skip this cell
        store.thing = expensive_operation()

    cell3:
        print(store.thing)
    """

    def __init__(self, filename="file_munch.pkl"):
        self._filename = str(filename)
        self._load()

    def _save(self):
        pickle_save(self._filename, self.toDict())

    def _load(self):
        filename = self._filename
        if local.path(self._filename).exists():
            try:
                d = pickle_load(filename)
                del d["_filename"]
                self.update(d)
            except Exception:
                info(f"Store {self._filename} was unloadable")

    def _set(self, key, value):
        with file_lunch_lock:
            super().__setitem__(key, value)
            self._save()

    def __setitem__(self, key, value):
        if key.startswith("_"):
            return super().__setitem__(key, value)
        self._set(key, value)

    def __setattr__(self, key, value):
        if key.startswith("_"):
            return super().__setattr__(key, value)
        self._set(key, value)

    def __delitem__(self, key):
        with file_lunch_lock:
            super().__delitem__(key)
            self._save()

    def __delattr__(self, item):
        with file_lunch_lock:
            super().__delattr__(item)
            self._save()

    def __call__(self, key, fn, _clear_store=False):
        if _clear_store:
            self.rm(key)
        if key not in self:
            self[key] = fn()
        return self[key]

    def rm(self, key):
        if key in self:
            self.__delitem__(key)


def counter(iterable):
    list_ = list(iterable)
    count = len(list_)
    for i, val in enumerate(list_):
        info(f"{i+1} of {count}")
        yield val


def plumbum_switches(app):
    """
    Extract switch values from a plumbum app.
    Usage
    """
    switches = {}
    for switch_name, switch_func in app._switches_by_name.items():
        try:
            val = getattr(app, switch_name)
            if not isinstance(val, (type(None), int, float, str)):
                val = None
        except:
            val = None
        switches[switch_name] = val

    return switches


def munch_deep_copy(src, klass_set={}):
    def _recurse(src):
        if isinstance(src, dict):
            dst = Munch()
            for k, v in src.items():
                dst[k] = _recurse(v)
            klass = type(src)
            if klass in klass_set:
                dst = klass(**dst)
            return dst
        elif isinstance(src, list):
            return [_recurse(elem) for elem in src]
        elif isinstance(src, tuple):
            return tuple([_recurse(elem) for elem in src])
        elif isinstance(src, (int, float, str, type(None))):
            return src
        else:
            raise TypeError(f"Unsupported type '{type(src)}' in munch_deep_copy")

    if not isinstance(src, (Munch, list)):
        raise TypeError(f"Unsupported root type '{type(src)}' in munch_deep_copy")

    return _recurse(src)


def munch_abbreviation_string(m):
    """
    Given a Munch, create a string that communicates the contents
    of the k,v pairs in a concise human-readable way.  E.g.

    m = Munch(
        my_special_value=0.1,
        something_else='funk',
        a_list=['joe','bob'],
    )

    becomes => msv_0.1-se_funk-al_joe.bob

    If any str(v) is longer than 32, it will be converted to hexdigest
    """
    parts = []
    for k, v in m.items():
        abbrev = "".join(map(lambda s: s[0], k.split("_")))
        v = ".".join(map(str, v)) if isinstance(v, list) else str(v)
        if len(v) > 32:
            v = hashlib.md5(v.encode()).hexdigest()
        parts += [f"{abbrev}_{v}"]
    return "-".join(parts)


def ispace(start, length, steps):
    """Integer version of linspace"""
    return np.linspace(start, length - 1, steps, dtype=int)


def expand_slice_to_list(slice_):
    return list(range(slice_.start or 0, slice_.stop or len(slice_), slice_.step or 1))


def easy_join(df1, df2, common_col):
    """
    The common case of joining two dataframes by a common key.
    Example:
        easy_join(fea_by_ch_df[0], filter_df[["peak_i", "pass_quality"]], "peak_i")

    Semantics:
        result = easy_join(table1, table2, column)
    in SQL is:
        select * from table1 left join table2 on table1.column = table2.column
    """
    return df1.set_index(common_col).join(df2.set_index(common_col)).reset_index()


# Numpy helpers
# -----------------------------------------------------------------------------------


@contextmanager
def np_no_warn(category=RuntimeWarning):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=category)
        yield


def np_safe_divide(a, b, default=0.0):
    """
    return a/b or default if b is zero

    Important usage note:
        If a and b are scalars then they will be converted up to an array
        but it's a weird array because it isn't a 1x1 but rather an un-indexable
        np scalar type array.
        a = np_safe_divide(1, 0, default=0.0)
        a[0]  # Will raise "IndexError: too many indices for array"
        float(a)  # What you want
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    a = a.astype(float)
    b = b.astype(float)
    return np.divide(
        a,
        b,
        out=np.full_like(a, default, dtype=float),
        where=(b != 0.0) & ~(np.isnan(b)),
    )


def np_array_same(a, b):
    return np.allclose(a, b, equal_nan=True)
    # return ((a == b) | (np.isnan(a) & np.isnan(b))).all()


def np_1d_end_pad(a, full_size):
    """
    Extend a 1 dimensional array with right-side padding.

    Eg:
    a = np.array([1, 2, 3])
    np_1d_end_pad(a, 5) == np.array([1, 2, 3, 0, 0])
    """
    assert a.ndim == 1
    assert full_size >= a.shape[0]
    extra = full_size - a.shape[0]
    return np.pad(a, ((0, extra),), mode="constant")


def np_arg_last_where(bool_arr):
    """
    Return the LAST index where something is true.
    return None if there is no such location
    """
    assert bool_arr.ndim == 1
    n_elems = bool_arr.shape[0]
    if n_elems > 0 and np.any(bool_arr):
        last_i = np.nanargmax(bool_arr[::-1])
        if 0 <= last_i < n_elems:
            return n_elems - last_i - 1
    return None


def np_arg_first_where(bool_arr):
    """
    Return the FIRST index where something is true.
    return None if there is no such location
    """
    assert bool_arr.ndim == 1
    n_elems = bool_arr.shape[0]
    if n_elems > 0 and np.any(bool_arr):
        first_i = np.nanargmax(bool_arr)
        if 0 <= first_i < n_elems:
            return first_i
    return None


def np_row_sort(mat, arg_sorted_rows=None):
    """
    Sorting a matrix by its rows requires a bit of indexing kung-fu.
    """
    assert mat.ndim == 2
    if arg_sorted_rows is None:
        arg_sorted_rows = np.argsort(mat, axis=1)

    assert arg_sorted_rows.shape == mat.shape

    a = np.arange(mat.shape[0])
    return mat[a[:, None], arg_sorted_rows]


def np_fn_along(fn, a, b, axis):
    """
    Apply fn with a braodcast on the given axis.
    Based on: https://stackoverflow.com/a/30032182

    Example:
        a = np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])
        b = np.array([2, 1])

        # You can't "a * b" because it would error with "broadcast together with shapes (2,3) (2,)"
        # But you can do this:
        c = np_fn_along(np.multiply, a, b, axis=0)
        assert np.all(c == np.array([[2, 4, 6], [4, 5, 6]]))

    """
    dim_array = np.ones((1, a.ndim), int).ravel()
    dim_array[axis] = -1
    b_reshaped = b.reshape(dim_array)
    return fn(a, b_reshaped)


def np_within(x, target, bounds):
    """Is x within within bounds (positive or negative) of expected"""
    return (x - target) ** 2 < bounds**2


def np_shift(arr, num, fill_value=np.nan):
    # From https://stackoverflow.com/a/42642326
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def np_safe_nanmedian(arr):
    # Like np.nanmedian except doesn't freak out if input is all nans
    if np.all(np.isnan(arr)):
        return np.nan
    else:
        return np.nanmedian(arr)


def np_safe_nanmean(arr):
    # Like np.nanmean except doesn't freak out if input is all nans
    if np.all(np.isnan(arr)):
        return np.nan
    else:
        return np.nanmean(arr)


def npf(arr):
    """
    'Num-Py Float' shortcut for making a float array from lists.
    Often used in tests.
    """
    return np.array(arr, dtype=float)


def arr(*args):
    """
    Like npf but *args to array
    """
    return np.array(args)


def np_rot_mat(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def np_scale_mat(x, y):
    return np.array([[x, 0.0], [0.0, y]])


def np_complex_mag(a):
    return np.real(a * np.conj(a))


def gaussian_1d(x, amp, mean, std, const):
    return amp * np.exp(-0.5 * (((x - mean) / std) ** 2)) + const


def fit_gaussian_1d(y, x=None):
    """
    Fit a 1d Gaussian to y.

    Typical usage:
        x = np.arange(len(data))
        gauss_params = fit_gaussian_1d(x=x, y=data)
        fitted_curve = gaussian(_x, *gauss_params)

    Remember: this is NOT fitting to the distribution of y,
    but rather assumes that y is a Gaussian shape. If you want
    to fit the *distribution* then convert the data to a histogram
    first. For example:
        # data is normally distributed
        _y, _x = np.histogram(data, bins=100)
        _x = _x[:-1]  # Eliminate the extra edge that historgram returns
        gauss_params = fit_gaussian_1d(x=_x, y=_y)

    Arguments:
        y: ndarray to fit
        x: over domain (same length as y). If x is not specified
           then it will create a range over the same length.

    Returns:
        The gaussian parameters in the same order that gaussian_1d takes:
        (amp, mean, std) so that you can call gaussian_1d as:
            gaussian_1d(x, *gauss_params)
    """

    from scipy import optimize  # Defer slow import

    if x is None:
        x = np.arange(len(y))

    n = len(y)
    assert len(x) == n

    const = np.min(y)
    amp = np.max(y)
    mean = x[np.argmax(y)]
    _y = y - const
    std = np.sqrt(np.sum(_y * (x - mean) ** 2) / np.sum(_y))

    popt, _ = optimize.curve_fit(
        gaussian_1d,
        x,
        y,
        p0=(amp, mean, std, const),
        bounds=(
            (0.0, np.min(x), 0.0, np.min(y)),
            (np.inf, np.max(x), np.inf, np.max(y)),
        ),
    )
    return tuple(popt)


def np_row_span_where_true(bool_mat):
    """
    For each row in the bool_mat find the length of the span
    where the values are True.  This is implemented with np.argmin()
    but that function has the annoyance that a row that is all True
    will return argmin==0 instead of the length of the row
    so this is compensated for with an additional np.all() call

    Example:
        a = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ])
        assert utils.np_row_span_where_true(a == 0).tolist() == [3, 2, 1, 0]  # The lengths of the rows
    """
    check.array_t(bool_mat, ndim=2, dtype=bool)
    amin = np.argmin(bool_mat, axis=1)
    amin[np.all(bool_mat, axis=1)] = bool_mat.shape[1]
    return amin


def np_choice(arr, size):
    iz = np.random.choice(arr.shape[0], size)
    return arr[iz]


def np_arg_find_closest_in_sorted_array(arr, val):
    # Be careful, sometimes you might want np.interp
    # Based on: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    check.array_t(arr, ndim=1)
    idx = np.searchsorted(arr, val, side="left")
    if idx > 0 and (
        idx == len(arr) or math.fabs(val - arr[idx - 1]) < math.fabs(val - arr[idx])
    ):
        return idx - 1
    else:
        return idx


def np_find_closest_in_sorted_array(arr, val):
    return arr[np_arg_find_closest_in_sorted_array(arr, val)]


def np_mult_along_axis(A, B, axis):
    """Adapted from https://stackoverflow.com/a/62655664"""

    # ensure we're working with Numpy arrays
    # A = np.array(A)
    # B = np.array(B)

    # shape check
    if axis >= A.ndim:
        raise AxisError(axis, A.ndim)
    if A.shape[axis] != B.size:
        raise ValueError(
            "Length of 'A' along the given axis must be the same as B.size"
        )

    # np.broadcast_to puts the new axis as the last axis, so
    # we swap the given axis with the last one, to determine the
    # corresponding array shape. np.swapaxes only returns a view
    # of the supplied array, so no data is copied unnecessarily.
    shape = np.swapaxes(A, A.ndim - 1, axis).shape

    # Broadcast to an array with the shape as above. Again,
    # no data is copied, we only get a new look at the existing data.
    B_brc = np.broadcast_to(B, shape)

    # Swap back the axes. As before, this only changes our "point of view".
    B_brc = np.swapaxes(B_brc, A.ndim - 1, axis)

    return A * B_brc


class DataclassUnpickleFromMunchMixin:
    def __setitem__(self, key: str, value: Any) -> None:
        """
        This is here for dataclasses that need to unpickle from a munch for backwards compatibility.

        pickle.load will call setitem here, because it thinks it's a munch, so we'll translate that to a setattr call.
        """
        # I realize that this is inefficient doing this on every setitem,
        # but there's only 20 ish fields so it's quick enough.
        field_dict = {f.name: f for f in dataclasses.fields(self)}
        if key not in field_dict:
            log.warning(
                "Attempted to set %s on %s, but the field isn't defined, so the value won't be set.",
                key,
                self.__class__.__name__,
            )
        else:
            super().__setattr__(key, value)


class timing:
    """Used in a with statement to time a block of code."""

    def __init__(self, label):
        self.start = None
        self.end = None
        self.label = label

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        self.end = time.time()
        print(f"{self.label}: {self.end-self.start}")
