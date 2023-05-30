"""
DEPRECATED

This is a tricky piece of memory tracking.

After much experimentation I found that the only good way to reliably
track memory usage was to use linux cgroups. Since things are already dockerized
I use the docker cgroups to track memory usage.

The trick is that root can over write the cgroup's memory.max_usage_in_bytes
file at any time and this trick can be used to get intermediate memory usage statistics.
However, this trick requires that the container have sudo access and have the
cgroup files mapped into the container.

Example docker run with the ability to sudo the cgroup, assuming that
the container is not running as root:

    sudo bash -c "echo 'root ALL=(ALL:ALL) ALL' > /tmp/sudoers"
    sudo bash -c "echo '$USER ALL=(ALL:ALL) NOPASSWD:ALL' >> /tmp/sudoers"
    docker run -it --rm \
        --volume /sys/fs/cgroup/memory:/cgroup_mem:rw \
        --volume /tmp/sudoers:/etc/sudoers \
        ...

Note the mapping of the cgroup into /cgroup_mem
"""

import gc
import re
import time
from contextlib import contextmanager

from plumbum import local


def _get_max_mem():
    """Return the current cgroup's memory high water mark."""
    try:
        with open("/sys/fs/cgroup/memory/memory.max_usage_in_bytes") as f:
            return float(f.read().strip())
    except Exception:
        return 0


def _get_container_id():
    """Parse the current running container's ID"""
    try:
        with open("/proc/self/cgroup") as f:
            for line in f.readlines():
                m = re.search(r"/docker/(.+)", line)
                if m:
                    return m.group(1)
    except Exception:
        return None


_stack = []

# The logs track the results from the blocks
mem_log = {}
duration_log = {}


def reset():
    global mem_log
    global duration_log
    mem_log = {}
    duration_log = {}


def _gc_collect():
    """mock-point"""
    gc.collect()


def _reset_mem_max(id):
    """mock-point"""
    local["sudo"](
        "--non-interactive",
        "bash",
        "-c",
        f"echo 0 > /cgroup_mem/docker/{id}/memory.max_usage_in_bytes",
    )


def _trace(name, duration, mem):
    """mock-point"""
    print(
        f"\u001b[1m{name}\u001b[0m "
        f"{duration:2.1f} sec "
        f"mem={mem / 1024 ** 3:2.2f} GB "
    )


@contextmanager
def mem_section(name, trace_fn=_trace):
    """
    Usage:
        with mem_section("step1"):
            something_that_eats_memory()

    Note that this allows nesting and as long as the container has
    sudo access to write to /cgroup_mem it will track sub-usage correctly.
    Example:
        with mem_section("all_steps"):
            with mem_section("step1"):
                something_that_eats_a_lot_of_memory()
            with mem_section("step2"):
                something_that_eats_a_little_memory()

        Prints:
            step1 1.0 sec mem=20.00 GB
            step2 1.0 sec mem=1.00 GB
            all_steps 2.0 sec mem=20.00 GB

    If the container does not have the correct sudo access it will
    not be able to reset the memory on each section and the result will be the same.
        Example from above with sudo access:
            step1 1.0 sec mem=20.00 GB
            step2 1.0 sec mem=20.00 GB       # <<< Note that this is 20, not 1
            all_steps 2.0 sec mem=20.00 GB

    This system garbage collects to clean things up before the block.

    You can fetch the memory and timing by the name with the mem_log and
    duration_log tables.

    with mem_section("all_steps", trace=False):
        with mem_section("step1", trace=False):
            something_that_eats_a_lot_of_memory()
        with mem_section("step2", trace=False):
            something_that_eats_a_little_memory()

    print(f"all_steps used {mem_log['all_steps']} bytes")
    """

    global _stack
    _gc_collect()

    mem = 0
    _stack.append(0)

    id = _get_container_id()
    if id is not None:
        try:
            _reset_mem_max(id)
        except Exception:
            pass

    start_time = time.time()

    yield

    end_time = time.time()

    if id is not None:
        mem = _get_max_mem()
    mem = max(mem, _stack.pop())
    if len(_stack) > 0:
        # Modify the last on the stack so it learns about
        # the max memory that was used in this nested section
        _stack[-1] = max(_stack[-1], mem)

    duration = end_time - start_time

    if trace_fn is not None:
        trace_fn(name, duration, mem)

    global mem_log
    mem_log[name] = mem

    global duration_log
    duration_log[name] = duration
