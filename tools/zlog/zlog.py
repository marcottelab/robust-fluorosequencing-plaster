"""
Usage patterns:

Initialization at main entrypoint:

    # Ensure that either:
    #   * logger.yaml is found parallel to the entrypoint
    #   * --OR-- that

    if __name__ == "__main__":
        from plaster.tools.zlog import zlog
        with zlog.app_start():
            do_stuff()

Initialization for each module follows Python standard:

    from logging import getLogger
    log = getLogger(__name__)
    log.debug("blah")

Understanding logging
    * If you do not initialize the python logger (ie do not call dictConfig)
      then you will see ALL logs to warning and error and NOTHING else
      to ALL modules.
    * If you add a root record in the logger and no other logger than
      you will see ALL message everywhere at the level enabled in the root
      logger because all modules are children of root so they will default to
      propagate up to the root and the root logger rules will apply.
    * Therefore it is a bad idea to set the root logger to anything other
      than WARNING because it will mean that we will see the the verbose debugging
      of every crazy module you forget to list in the config dict.

"""

import warnings

warnings.warn(
    "zlog is deprecated, please migrate to structlog", DeprecationWarning, stacklevel=2
)

import inspect
import logging
import logging.config
import os
import re
import sys
import threading
import traceback
from contextlib import contextmanager

import numpy as np
from icecream import ic
from plumbum import colors, local
from plumbum.commands.modifiers import ExecutionModifier
from pythonjsonlogger.jsonlogger import JsonFormatter

# Ths tb_pat regular expression is used to break up traceback for colorization

tb_pat = re.compile(r"^.*File \"([^\"]+)\", line (\d+), in (.*)")


def human_readable_type_and_value(arg):
    """
    Examine the type of arg and emit it in friendly ways.
    The set of types that are given special consideration is
    likely to grow. At moment, it only includes numpy but will
    likely containe special cases for pandas or other library types
    that we need special type info.
    """

    type_str = str(type(arg).__name__)
    val_str = str(arg)

    if isinstance(arg, np.ndarray):
        dims = ",".join([f"{dim}" for dim in arg.shape])
        type_str = f"ndarray[{dims}]{{{arg.dtype}}}"
        val_str = str(arg)
        if "\n" in val_str:
            # On multiline, add an extra newline to keep first line with remainder
            val_str = "\n" + val_str

    return type_str, val_str


def colorful_exception(error=None, formatted=None, msg=""):
    """
    Return a string of colorful easy-to-read version of an exception traceback.
    """

    s = ""

    def _traceback_match_filename(line):
        is_libs = False
        m = tb_pat.match(line)
        if m:
            file = m.group(1)
            lineno = m.group(2)
            context = m.group(3)
            real_path = os.path.realpath(file)
            relative_path = os.path.relpath(real_path)

            root = os.environ.get("ERISYON_ROOT")
            if root is not None:
                is_libs = True
                if real_path.startswith(root):
                    relative_path = re.sub(r".*/" + root, "./", real_path)
                    is_libs = False

            if real_path == __file__:
                # Any functions in the log helper are "libs"
                is_libs = True

            # Treat these long but commonly occurring path differently
            if "/site-packages/" in relative_path:
                relative_path = re.sub(r".*/site-packages/", ".../", relative_path)
            if "/dist-packages/" in relative_path:
                relative_path = re.sub(r".*/dist-packages/", ".../", relative_path)

            leading, basename = os.path.split(relative_path)
            # if leading and len(leading) > 0:
            #     leading = f"{'./' if leading[0] != '.' else ''}{leading}"
            return leading, basename, lineno, context, is_libs

        return None

    s += colors.red & colors.bold | error.__class__.__name__
    s += "('"
    error_message = str(error).strip()
    if error_message != "":
        s += colors.red | error_message
    s += "')\n"
    s += f"  was raised with message: "
    s += colors.red | f"'{msg}'\n"

    if formatted is None:
        formatted = traceback.format_exception(
            etype=type(error), value=error, tb=error.__traceback__
        )

    lines = []
    for line in formatted:
        lines += [sub_line for sub_line in line.strip().split("\n")]

    is_libs = False
    for line in lines[1:-1]:
        split_line = _traceback_match_filename(line)
        if split_line is None:
            # The is_libs flag is set every other line and remains
            # set for the followig line which shows the context
            s += (colors.DarkGray if is_libs else colors.white) | line + "\n"
        else:
            leading, basename, lineno, context, is_libs = split_line
            s += "  "
            if is_libs:
                s += colors.DarkGray | leading + "/ "
                s += colors.DarkGray | basename + ":"
                s += colors.DarkGray | lineno + " in function "
                s += colors.DarkGray | context + "\n"
            else:
                s += colors.yellow | leading + "/ "
                s += colors.yellow & colors.bold | basename
                s += ":"
                s += colors.yellow & colors.bold | lineno
                s += " in function "
                s += colors.magenta & colors.bold | context + "\n"

    return s


class ColorfulFormatter(logging.Formatter):
    """
    A formatter that adds color, especially to exceptions
    but also to any record that contains spy fields (see def spy)
    """

    def format(self, record: logging.LogRecord) -> str:
        type_str, val_str = human_readable_type_and_value(record.msg)

        if hasattr(record, "spy_variable_name"):
            # This record contains spy variable fields which are
            # colorized differently. See def spy() to see how these fields
            # go into this record.
            msg = (
                (colors.green | f"{record.spy_variable_name}")
                + (colors.DarkGray | ":")
                + (colors.yellow | type_str)
                + (colors.DarkGray | " = ")
                + (colors.white | val_str)
            )
        else:
            msg = colors.green | str(record.msg)

        if hasattr(record, "exc_info") and record.exc_info is not None:
            # The case of an exception use the fancy colorful exception formatter
            e = record.exc_info[1]
            formatted = traceback.format_exception(
                etype=type(e), value=e, tb=e.__traceback__
            )
            msg = colorful_exception(error=e, formatted=formatted, msg=record.msg)
        else:
            name = ""
            include_record_name = False
            if include_record_name:
                name = colors.cyan | f"{record.name} "

            filename = record.filename
            if filename.startswith("<ipython"):
                filename = "notebook_cell"
            msg = name + (colors.blue | f"{filename}:{record.lineno}] ") + msg

        return msg


ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class TypeAwareJsonFormatter(JsonFormatter):
    """
    Adds human readable type information to json log messages when the spy()
    function adds the "spy_variable_name" to the log record
    """

    def format(self, record: logging.LogRecord) -> str:
        if hasattr(record, "spy_variable_name"):
            verbose_type_str, val_str = human_readable_type_and_value(record.msg)
            record.msg = f"{record.spy_variable_name}:{verbose_type_str} = {val_str}"

        # STRIP escape code
        if isinstance(record.msg, str):
            record.msg = ansi_escape.sub("", record.msg)

        return super().format(record)


class ZlogFG(threading.Thread, ExecutionModifier):
    """
    Used in place of plumbum.FG, logs stdout and stderr of subprocess to logger

    Adapted from https://codereview.stackexchange.com/questions/6567/redirecting-subprocesses-output-stdout-and-stderr-to-the-logging-module
    """

    def __init__(
        self,
        logger,
        level=logging.INFO,
        retcode=0,
        timeout=None,
        drop_dup_lines=False,
        convert_to_cr=False,
    ):
        threading.Thread.__init__(self)
        self.logger = logger
        self.level = level
        self.retcode = retcode
        self.timeout = timeout
        self.drop_dup_lines = drop_dup_lines
        self.convert_to_cr = convert_to_cr
        self.daemon = False
        self.level = level
        self.fd_read, self.fd_write = os.pipe()
        self.pipe_reader = os.fdopen(self.fd_read)
        self.start()

    def fileno(self):
        """Return the write file descriptor of the pipe"""
        return self.fd_write

    def run(self):
        """Run the thread, logging everything."""
        last_line = None
        for line in iter(self.pipe_reader.readline, ""):
            if self.drop_dup_lines and line == last_line:
                continue
            last_line = line

            logging.log(self.level, line.strip("\n"))
            if local.env.get("ERISYON_HEADLESS") != "1":
                if self.convert_to_cr:
                    line = line.strip("\n") + "\r"
                sys.stdout.write(line)

        self.pipe_reader.close()

    def close(self):
        """Close the write end of the pipe."""
        os.close(self.fd_write)

    def __rand__(self, cmd):
        try:
            cmd(
                retcode=self.retcode,
                stdin=None,
                stdout=self,
                stderr=self,
                timeout=self.timeout,
            )
        finally:
            self.close()


def metrics(message, log=None, **kwargs):
    warnings.warn("metrics is deprecated", DeprecationWarning, stacklevel=2)

    # leaving this in place to not break old reports
    pass


def spy(*args, log=None, np_save=None):
    warnings.warn(
        "spy is deprecated, please use icecream.ic directly",
        DeprecationWarning,
        stacklevel=2,
    )
    from icecream import ic

    ic(*args)


def tell(*args, log=None):
    """
    tell() is a wrapper for print().

    print() is HARMFUL because:
        * When used as a "one-off debugging" it doesn't print context info (file:line)
          and thus when such a print() is accidentally left in the code base they can be very
          hard to find and kill.  (This is especially painful when the print has no
          identifier like print(foo) which results in a random number like "1.23" being
          emitted in some random state.
          Use spy() for casual print debugging, not tell()
        * When used for debugging, you often want variable name AND type info. See spy()
        * In some cases print() isn't pushed through the log system so that when a module is run on
          a headless instance the print() may get lost.  When there is a "stdio capture"
          log as is common with some services these print() statements end up emitting into
          the logs without context (level, module, source, etc)
        * When decorating the prints with console color those escape characters end up
          pushed into any capture-level logs making them hard to read.

    But yet there are times when you want to TELL a CLI user something and you want to
    print this WITHOUT log headers and you would like that to go in to the logs ALSO
    without having call both print(...) and log.info(...) -- this is the intended use-case
    for tell().

    Behavior of tell():
        * print message to stdout without additional headers and preserving escape sequences
        * ALSO write the message to the specified log (or "plaster.zlog.tell" logger if not specified)
          with full trace information and strip escape code.
        * If an annoying tell() gets left in you can either look in the logs
          or set "find_annoying_print = True" below to hunt it down.
        * If the environ "TELL_SHOW_CONTEXT" == "1" then it also prints file:line content.
    """

    find_annoying_print = False
    if find_annoying_print or local.env.get("TELL_SHOW_CONTEXT") == "1":
        frame = inspect.currentframe()
        back_frame = frame.f_back
        sys.stdout.write(
            colors.cyan | f"[{back_frame.f_code.co_filename}:{back_frame.f_lineno}]: "
        )

    is_jupyter_context = "ipykernel_launcher.py" in sys.argv[0]

    if local.env.get("ERISYON_HEADLESS") != "1" or is_jupyter_context:
        # print in jupyter context even in headless because it will be trapped
        # and we want all output visible in jupyter
        print(*args, flush=True)

    if log is None:
        log = logging.getLogger("plaster.zlog.tell")

    for arg in args:
        if isinstance(arg, str):
            log.info(ansi_escape.sub("", arg))
        else:
            log.info(str(arg))


def important(*args, log=None):
    """
    Like tell(), but adds emphasis.
    """
    colorful_args = [(colors.yellow | arg) for arg in args]
    tell(*colorful_args, log=log)


@contextmanager
def add_log_fields(**kwargs):
    # leaving this in place in preent breaking any callers
    yield


class HandlerHook(logging.Handler):
    """
    Wraps a handler to capture its messages and then super().handle them
    """

    def __init__(self, handler_to_copy, forward_to_handler, *args, **kwargs):
        self.handler_to_copy = handler_to_copy
        self.forward_to_handler = forward_to_handler
        super().__init__(*args, **kwargs)

    def handle(self, record: logging.LogRecord) -> None:
        self.handler_to_copy.handle(record)
        self.forward_to_handler.handle(record)


def add_handler(handler: logging.Handler, logger_name=None):
    """
    Add a handler to all existing loggers

    Arguments:
        handler: A new handler to add to the loggers
        logger_name: Which logger to add to or None for all

    Example:
        formatter = TypeAwareJsonFormatter(format="%(name)s %(asctime)s %(levelname)s %(message)s %(filename)s %(lineno)d")
        handler = logging.StreamHandler(open(per_run_log_path, "w"))
        handler.setFormatter(formatter)
        add_handler(handler, logger_name="plaster.zlog.profile")
    """

    assert isinstance(handler, logging.Handler)

    if logger_name is None:
        loggers = [
            logging.getLogger(log_name)
            for log_name in logging.Logger.manager.loggerDict.keys()
        ]
    else:
        loggers = [logging.getLogger(logger_name)]

    for logger in loggers + [logging.root]:
        logger.addHandler(handler)


config_dict = None


@contextmanager
def app_start(path_to_yaml=None):
    # keeping this here to not break callers which expect it to exist as a contextmanager
    from plaster import env

    env.configure_logging()

    import structlog

    log = structlog.get_logger().bind(path_to_yaml=path_to_yaml)

    try:
        yield
    except Exception as e:
        log.exception("in app_start")


def notebook_start(path_to_yaml=None):
    # keeping notebook_start in place to not break old reports which expect its presence
    from plaster import env

    env.configure_logging()


def terminal_size():
    import fcntl
    import struct
    import termios

    try:
        h, w, hp, wp = struct.unpack(
            "HHHH", fcntl.ioctl(0, termios.TIOCGWINSZ, struct.pack("HHHH", 0, 0, 0, 0))
        )
        return w, h
    except OSError:
        # This can happen in a containerized context where no console exists
        # 80, 80 chosen as reasonable default w, h for imaginary terminal
        return 80, 80


def is_headless():
    """Mock-point"""
    return os.environ.get("ERISYON_HEADLESS", "0") == "1"


def _input():
    """Mock-point"""
    return input()


def input_request(message, default_when_headless):
    """
    Ask the user for input, but if headless return default_headless.
    If default_headless is an exception, then raise that.

    Note that this REQUIRES a default_headless argument so that you
    can not be lazy and avoid the question about what should happen in the
    case of a headless run.

    If this should never happen when headless, then pass in
    an Exception to the default_when_headless and it will be raised.
    """
    if is_headless():
        if isinstance(default_when_headless, Exception):
            raise default_when_headless
        return default_when_headless

    # Do not use the input(message) here because it doesn't wrap properly
    # in some cases (happened when I was using escape codes, but I didn't
    # bother to figure out why.)... Just print() works fine.
    tell(message)
    return _input()


def confirm_yn(message, default_when_headless):
    """
    Prompt the user for an answer.
    See input_request() to understand why this requires a default_headless argument.
    """
    resp = None
    while resp not in ("y", "n"):
        resp = input_request(message + "(y/n): ", default_when_headless)
        resp = resp.lower()[0]
    return resp == "y"


def h_line(marker="-", label=""):
    count = (terminal_size()[0] - 1) // len(marker) - len(label)
    return label + marker * count
