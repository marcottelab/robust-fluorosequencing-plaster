"""Environmental concerns

This module provides a central place to add runtime environmental concerns.
"""
import logging
import logging.config
import sys
from enum import Enum

import structlog


class EnvironmentKind(Enum):
    UNKNOWN = 1
    NOTEBOOK = 2
    IPYTHON_TERM = 3
    PYTHON_TERM = 4


def which_env_kind() -> EnvironmentKind:
    """Detect whether you're in a notebook environment

    Adapted from: https://stackoverflow.com/a/39662359
    """
    try:
        # get_ipython is available in a notebook context
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return EnvironmentKind.NOTEBOOK  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return EnvironmentKind.IPYTHON_TERM  # Terminal running IPython
        else:
            return EnvironmentKind.UNKNOWN  # Other type (?)
    except NameError:
        if sys.stderr.isatty():
            return EnvironmentKind.PYTHON_TERM  # Definitely a Python interpreter

        return EnvironmentKind.PYTHON_TERM  # Probably standard Python interpreter


def configure_logging(
    *,
    env_kind: EnvironmentKind = EnvironmentKind.UNKNOWN,
    level: int = logging.INFO,
):
    """Configure our logging subsystem

    Since this controls logging, you probably want to call this as early as possible.

    This function takes into account whether we're attached to an interactive prompt to
    determine which set of processors to utilize.

    It assumes Notebook, and iPython terminals are user friendly and outputs in text.
    Potentially headless environments, including `python -m foo`, output in json.

    The logging setup is as documented here:
    https://www.structlog.org/en/stable/standard-library.html#rendering-using-structlog-based-formatters-within-logging

    The outcome of this configuration is that libraries which log using stdlib's logging
    will have their log lines handled by structlog; the logger we choose to use at runtime.

    Args
    ----
    env_kind
        The kind of environment; defaults to trying to guess via `which_env_kind`
    level
        The minimum log level to output at; presumes this is a valid value from stdlib
        logging
    """

    # These processors are shared by both of:
    #  - structlog originating events
    #  - stdlib log events; typically from libraries we may include
    shared_processors = [
        # Add log level to event dict.
        structlog.stdlib.add_log_level,
        # Add the name of the logger to event dict.
        structlog.stdlib.add_logger_name,
        # Perform %-style formatting.
        structlog.stdlib.PositionalArgumentsFormatter(),
        # If the "stack_info" key in the event dict is true, remove it and
        # render the current stack trace in the "stack" key.
        structlog.processors.StackInfoRenderer(),
        # If some value is in bytes, decode it to a unicode str.
        structlog.processors.UnicodeDecoder(),
        # Add callsite parameters.
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
    ]

    # these processors are only applied to log events originating from structlog; typically plaster
    processors = [
        # If log level is too low, abort pipeline and throw away log entry.
        structlog.stdlib.filter_by_level,
    ] + shared_processors

    # these processors are only applied to log events originating from stdlib logging, likely libs
    # these are passed as the foreign pre chain to ProcessorFormatter which then includes the
    # logging parameters we expect into the log record; essentially making log lines from stdlib
    # and structlog analogous
    stdlib_processors = [
        # Add extra attributes of LogRecord objects to the event dictionary
        # so that values passed in the extra parameter of log methods pass
        # through to log output.
        structlog.stdlib.ExtraAdder(),
    ] + shared_processors

    if env_kind is EnvironmentKind.UNKNOWN:
        env_kind = which_env_kind()

    # The default renderer when in an interactive session; this outputs color by default
    # if we end up renderering these to a location that cannot handle color escape chars
    # we need to detect that and pass colors=False to this constructor
    renderer = structlog.dev.ConsoleRenderer()

    if env_kind in (EnvironmentKind.NOTEBOOK, EnvironmentKind.IPYTHON_TERM):
        # Pretty printing when we run in a terminal session.
        processors.extend(
            [
                structlog.dev.set_exc_info,
            ]
        )

    if env_kind in (EnvironmentKind.PYTHON_TERM, EnvironmentKind.UNKNOWN):
        # Print JSON when we run in a headless environment
        # Also print structured tracebacks.
        processors.extend(
            [
                # If the "exc_info" key in the event dict is either true or a
                # sys.exc_info() tuple, remove "exc_info" and render the exception
                # with traceback into the "exception" key.
                structlog.processors.format_exc_info,
                structlog.processors.TimeStamper("iso"),
                structlog.processors.dict_tracebacks,
            ]
        )

        stdlib_processors.extend(
            [
                structlog.processors.TimeStamper("iso"),
            ]
        )

        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    #
    # stdlib logging
    #
    def extract_from_record(_, __, event_dict):
        """move parameters from log record to event_dict"""

        # this block is lifted from the structlog docs; chosen for inclusion here
        # since I expect it'll capture joblib and zap correctly
        if record := event_dict.get("record"):
            event_dict["thread_name"] = record.threadName
            event_dict["process_name"] = record.processName

        return event_dict

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        # Move parameters from the log record to the event_dict
                        extract_from_record,
                        # Removes the metadata that structlog adds to lines to determine their source
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        # output handler is the final processor entry
                        renderer,
                    ],
                    "foreign_pre_chain": stdlib_processors,
                }
            },
            "handlers": {
                "default": {
                    "level": logging.getLevelName(level),
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "loggers": {
                "": {  # this is the root logger in python
                    "handlers": ["default"],
                    "level": logging.getLevelName(level),
                    "propagate": True,
                }
            },
        }
    )
