import gc
import logging
import sys
import time
from contextlib import contextmanager
from os.path import realpath

import numpy as np
from munch import Munch
from plumbum import colors, local

from plaster.tools.utils import utils
from plaster.tools.zlog import zlog

log = logging.getLogger(__name__)


class Progress:
    stack = []
    suppress_progress = False

    def __init__(self, description, n_total=None, log=None):
        self.description = description
        self.start_time = None
        self.stop_time = None
        self.n_complete = None
        self.n_total = n_total
        self.n_renders = 0
        self.log = log

    @classmethod
    @contextmanager
    def quiet(cls):
        old_suppress_progress = cls.suppress_progress
        cls.suppress_progress = True
        yield
        cls.suppress_progress = old_suppress_progress

    def __enter__(self):
        self.start_time = time.time()
        self.stack += [self]
        self.render()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_time = time.time()
        self.render()
        self.stack.pop()

    def get_msg(self, extra_info=None):
        indent = 4
        state = (" " * indent) + self.description + ": "
        if self.stop_time is not None:
            elapsed = self.stop_time - self.start_time
            state += f"Finished in {utils.elapsed_time_in_minutes_seconds(elapsed)}"
        else:
            if self.n_complete is None:
                state += "Initializing"
            else:
                if self.n_total is None:
                    state += f"{self.n_complete} of unknown"
                else:
                    bar_length = 30
                    done = int(
                        bar_length * float(self.n_complete) / float(self.n_total)
                    )
                    not_done = bar_length - done
                    remain_char = "\u2591"
                    done_char = "\u2587"
                    state += f"{done_char * done}{remain_char * not_done}"

                if self.n_complete > 0:
                    elapsed = time.time() - self.start_time
                    remain = (elapsed / self.n_complete) * (
                        self.n_total - self.n_complete
                    )

                    state += (
                        f" elapsed: {utils.elapsed_time_in_minutes_seconds(elapsed)},"
                        f" ~remain: {utils.elapsed_time_in_minutes_seconds(remain)}   "
                    )

            if extra_info is not None:
                state += " " + extra_info

        return state

    def render(self, extra_info=None):
        if (
            local.env.get("ERISYON_HEADLESS") != "1"
            and local.env.get("ERISYON_SUPPRESS_PROGRESS_BARS") != "1"
            and not self.suppress_progress
        ):
            clear_to_eol = "\x1b[K"

            if self.n_renders > 0:
                up_line = "\x1b[A"
                sys.stdout.write(up_line)

            self.n_renders += 1

            # This terminates in "\n" so that any traces that arrive during
            # the progress will add a new line. Note above logic for up_line
            sys.stdout.write("\r" + self.get_msg(extra_info) + clear_to_eol + "\n")
            # if self.stop_time is not None:
            #     sys.stdout.write("\n")

    def __call__(self, n_complete, n_total, extra_info="", **kwargs):
        self.n_complete = n_complete
        self.n_total = n_total
        self.render(extra_info)
