import inspect
import logging
import os

import psutil

log = logging.getLogger(__name__)

vms_last = None
vms_pid = None


def vms_prof(msg=""):
    """A quick and dirty mem profiler."""
    global vms_last, vms_pid

    process = psutil.Process(os.getpid())
    mem = process.memory_info().vms
    used = 0
    gb = 2**30

    if vms_pid is None or vms_pid != os.getpid():
        vms_pid = os.getpid()
        vms_last = mem
    else:
        used = mem - vms_last

    frame = inspect.currentframe()
    try:
        context = inspect.getframeinfo(frame.f_back)
        line = (
            f"Used {used / gb:+.2f} GB {os.path.basename(context.filename)}:"
            f"{context.lineno}] {msg}\n"
        )
        log.info(line)

    finally:
        del frame
        vms_last = None
        if os.getpid() == vms_pid:
            process = psutil.Process(os.getpid())
            vms_last = process.memory_info().vms
        else:
            vms_last = 0
