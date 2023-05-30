import logging
import pathlib


def location() -> pathlib.Path:
    """Return the path to Plaster directory."""

    return pathlib.Path(__file__).parent.absolute()


def cached_dir() -> pathlib.Path:
    """Return the path to cached data directory."""

    logging.warning("plaster.root.cached_dir() currently only valid in development")
    return location().parent.parent / ".cached_data"
