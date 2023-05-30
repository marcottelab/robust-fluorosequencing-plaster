"""

Example of using profile in-place

    prof_start_record_capture()

    with prof("foo"):
        time.sleep(1)

    profile_lines = prof_stop_record_capture()
    profile_dump(profile_lines)
"""


import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager

from munch import Munch
from plumbum import colors

from plaster.tools.utils import utils
from plaster.tools.zlog.zlog import tell

prof_records = None


def prof_start_record_capture():
    """
    This is a very unusual function used for unusual testing circumstances.
    This is probably not the function you are looking for.
    See prof_start() and prof().

    This is used in cases of notebooks and some tests where you want
    to grab the profile records during a run.
    """
    global prof_records
    prof_records = []


def prof_stop_record_capture():
    global prof_records
    will_return = prof_records
    prof_records = None
    return will_return


def prof_clear_memory_highwater_mark():
    try:
        with open("/cgroup_mem/memory.max_usage_in_bytes", "w") as f:
            f.write("0")
    except Exception as e:
        pass


def prof_get_memory_highwater_mark():
    try:
        with open("/cgroup_mem/memory.max_usage_in_bytes", "r") as f:
            return int(f.read())
    except Exception as e:
        return 0


def prof_start(
    block_name,
    log=None,
    stacklevel=2,
    _group_children_by=None,
    _tell=False,
    **kwargs,
):
    global prof_stack

    if log is None:
        log = logging.getLogger("plaster.zlog.profile")

    full_name = ".".join([p[0] for p in prof_stack] + [block_name])

    now = time.time()
    prof_clear_memory_highwater_mark()
    prof_stack += [(block_name, full_name, now, log, _group_children_by, _tell, kwargs)]
    log.info(f"{full_name} start.", stacklevel=stacklevel)


def prof_stop(stacklevel=2):
    block = prof_stack.pop()
    name, full_name, start, log, _group_children_by, _tell, kwargs = block

    now = time.time()
    elapsed = now - start
    mem = prof_get_memory_highwater_mark()
    mem_gb = mem / 1024**3

    kwargs_per_sec = {f"{key}_per_sec": val / elapsed for key, val in kwargs.items()}

    kwargs_strs = [
        f"{key}={val:2.2f}" if isinstance(val, (int, float)) else f"{key}={val}"
        for key, val in {**kwargs, **kwargs_per_sec}.items()
    ]

    record = Munch(
        name=full_name,
        elapsed=elapsed,
        mem_gb=mem_gb,
        group_children_by=_group_children_by,
        **kwargs,
        **kwargs_per_sec,
    )

    msg = f"{full_name} stop. secs={elapsed:2.2f} mem={mem_gb:2.1f} {' '.join(kwargs_strs)}"
    log.info(
        msg,
        extra=dict(
            plaster_profile=json.dumps(record),
        ),
        stacklevel=stacklevel,
    )

    global prof_records
    if prof_records is not None:
        prof_records += [record]

    if _tell:
        tell(msg)


@contextmanager
def prof(block_name, log=None, **kwargs):
    try:
        prof_start(block_name, log=log, stacklevel=4, **kwargs)
        yield
    finally:
        prof_stop(stacklevel=4)


prof_stack = []


def profile_from_string(log_str):
    remove_fields = (
        "name",
        "asctime",
        "filename",
        "levelname",
        "message",
        "plaster_profile",
        "lineno",
        "module",
        "process",
    )

    profile_lines = []
    for line in log_str.split("\n"):
        try:
            log_dict = json.loads(line)
            if log_dict["name"] == "plaster.zlog.profile":
                if "plaster_profile" in log_dict:
                    plaster_profile_dict = json.loads(log_dict["plaster_profile"])
                    for fld in remove_fields:
                        log_dict.pop(fld, None)
                    profile_lines += [
                        Munch.fromDict(dict(**plaster_profile_dict, **log_dict))
                    ]
        except json.JSONDecodeError:
            pass

    return profile_lines


def profile_from_file(log_file):
    log_str = utils.load(log_file, return_on_non_existing="")
    return profile_from_string(log_str)


def _profile_dump_print(*args, **kwargs):
    """Mock-point"""
    print(*args, **kwargs)


def profile_dump(profile_lines):
    """
    There's no guarantee that profile lines are sorted correctly as
    they may be coming from different processes.

    Since the order is not guaranteed we have to
    do a lookup by name and then convert it into the desired tree

    Convert records like "foo.bar" to
      {
        "foo": {
          "bar": {
            "_records": [ {...}, {...} ],
          }
          "_records": [ {...}, {...} ],
        }
        "_records": [ {...}, {...} ],
      }

      Note that _records is a list of EVERY record that was
      associated with "foo.bar" which may or may not be grouped

    Example with grouping
      {
        "foo": {
          "prepare" {
            "_records": [
              {"elapsed": 0.01},
              {"elapsed": 0.02},
            ],
          }
          "analyze_field": {
            "_records": [
              {"elapsed": 5.4, "group_children_by": "field_i"},
            ],
            "align": {
              "_records": [
                {"field_i": 0, "elapsed": 1.2},
                {"field_i": 1, "elapsed": 2.3},
              ],
              "resample": {
                "_records": [
                  {"field_i": 0, "elapsed": 0.1},
                  {"field_i": 1, "elapsed": 0.2},
                ],
              }
            }
          }
          "_records": [ {"elapsed": 10.3} ],
        }
        "_records": [],
      }

      DESIRED OUTPUT:
        foo  elapsed=10.3
          prepare  elapsed=0.01
          prepare  elapsed=0.02
          analyze_field  elapsed=5.4  (groupby field_i)
            align (field_i=0)  elapsed=5.4
              resample  elapsed=0.1
            align (field_i=1)  elapsed=5.4
              resample  elapsed=0.2
    """

    def DictTree():
        def the_tree():
            return defaultdict(the_tree)

        return the_tree()

    max_name_len = 0
    tree = DictTree()
    for pline in profile_lines:
        parent = tree
        parts = pline.name.split(".")
        max_name_len = max(max_name_len, len(parts[-1]))
        for part in parts[:-1]:
            parent = parent[part]

        if "_records" not in parent[parts[-1]]:
            parent[parts[-1]]["_records"] = []
        parent[parts[-1]]["_records"] += [pline]

    special_fields = {"name", "elapsed", "group_children_by"}

    def _recurse(node, depth, group, group_val, parent_time):
        indent = "".join(["  "] * (depth - 1))
        children = [val for key, val in node.items() if key != "_records"]

        def _print_record(record):
            if not record["name"]:
                return

            show_key_vals = {}
            for key, val in record.items():
                if key not in special_fields:
                    if isinstance(val, float):
                        val = f"{val:2.2f}"
                    show_key_vals[key] = val

            show_key_vals = " ".join(
                [
                    (colors.green | f"{key}") + "=" + (colors.light_magenta | f"{val}")
                    for key, val in show_key_vals.items()
                ]
            )
            group_by = ""
            _group = record.get("group_children_by")
            if _group is not None:
                group_by = f"(group children by {_group})"
            elapsed = record.get("elapsed", 0)
            percent = f"{100 * elapsed / parent_time:>6.2f}%" if parent_time > 0 else ""
            _profile_dump_print(
                colors.cyan
                | f"{indent}{record['name'].split('.')[-1]: <{max_name_len}}",
                colors.yellow | f"{elapsed:>6.2f}",
                colors.bold & colors.yellow | percent,
                show_key_vals,
                group_by,
            )

        records = node.get("_records", [{"name": ""}])

        for record in records:
            if group is not None and group_val != record.get(group):
                continue

            _print_record(record)
            elapsed = record.get("elapsed", 0)

            _group = record.get("group_children_by")
            _group_vals = None
            if _group is not None:
                _group_vals = {
                    record.get(_group)
                    for child in children
                    for record in child["_records"]
                    if record.get(_group) is not None
                }

            for child in children:
                if _group is not None:
                    for group_val in sorted(_group_vals):
                        _recurse(child, depth + 1, _group, group_val, elapsed)
                else:
                    _recurse(child, depth + 1, group, group_val, elapsed)

    _recurse(tree, 0, None, None, None)
