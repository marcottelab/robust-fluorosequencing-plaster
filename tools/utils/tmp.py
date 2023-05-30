"""
This wrapper is meant for all temporary directories and files
in the Erisyon context. It uses ERISYON_TMP to know where to
locate files. This is especially important because ERISYON_TMP
folder is carefully configured in the docker containers
and you must not use the /tmp for this reason.

Also, it allows creating tmp_files inside of a tmp_folder context
without changing the current working directory.

This also consolidates all caching into $ERISYON_TMP/cache.

Example:

    from plaster.tools.utils.tmp import tmp_folder, tmp_file
    with tmp_folder():
        # The folder has been made under $ERISYON_TMP
        with tmp_file() as my_file:
            # my_file will be created IN THE TMP FOLDER that was just created!
            save(foo, my_file)

    with tmp_file() as my_file:
        # my_file will be created under $ERISYON_TMP
        save(foo, my_file)

    with tmp_file(prefix="foo") as my_file:
        # my_file will be created under $ERISYON_TMP with a "foo" prefix
        save(foo, my_file)

    with tmp_folder(chdir=True):
        # my_file will be created in the temporary folder
        save(foo, my_file)

"""
import hashlib
import os
import pickle
import tempfile
from contextlib import contextmanager
from logging import getLogger

from plumbum import local

log = getLogger(__name__)

_current_tmp_folder = None


def erisyon_tmp():
    # 1/25/2021: DHW - ZBS added this at some point ostensibly to improve some dev cycle. It never expires so it's causing issues for people using controlpanel.
    # 1/25/2021: DHW commented this out
    # if local.env.get("ON_AWS"):
    #     _erisyon_tmp = "/erisyon/jobs_folder/zack/_tmp"
    # else:

    _erisyon_tmp = local.env.get("ERISYON_TMP")
    if _erisyon_tmp is None:
        raise FileNotFoundError("The ERISYON_TMP variable is not set")
    p = local.path(_erisyon_tmp)
    p.mkdir()
    return p


@contextmanager
def tmp_folder(remove=True, prefix=None, chdir=False):
    """
    See examples above.
    """
    global _current_tmp_folder
    old_current_tmp_folder = _current_tmp_folder
    tmp_path = local.path(tempfile.mkdtemp(dir=str(erisyon_tmp()), prefix=prefix))
    orig_cwd = local.cwd
    try:
        _current_tmp_folder = tmp_path
        if chdir:
            local.cwd.chdir(tmp_path)
        yield tmp_path
    finally:
        if chdir:
            local.cwd.chdir(orig_cwd)
        if remove:
            try:
                tmp_path.delete()
            except OSError:
                log.info(f"Unable to delete tmp folder {tmp_path}. Ignoring.")

        _current_tmp_folder = old_current_tmp_folder


@contextmanager
def tmp_file(remove=True, prefix=None):
    """
    If in the context of a tmp_folder, uses that
    otherwise write into erisyon_tmp.
    See examples above.
    """
    if _current_tmp_folder is None:
        folder = erisyon_tmp()
    else:
        folder = _current_tmp_folder

    fd, tmp_path = tempfile.mkstemp(dir=str(folder), prefix=prefix)
    os.close(fd)
    tmp_path = local.path(tmp_path)
    try:
        yield tmp_path
    finally:
        if remove:
            tmp_path.delete()


def cache_folder():
    """Get the default cache folder, do not create a hash of the arguments like cache_path"""
    dst_path = erisyon_tmp() / "cache"
    dst_path.mkdir()
    return dst_path


def cache_path(prefix, *args, **kwargs):
    """
    Make a path by concatenating prefix and an md5 of the args, kwargs

    Usage:
        some_argument_to_hash = 1
        found_cache, dst_path = cache_path("plaster_s3", some_argument_to_hash)
        if not found_cache:
            do_something_to_set_dst_path(dst_path)
        do_something_to_use_dst_path(dst_path)
    """

    hash_key = hashlib.md5(
        pickle.dumps(args, protocol=4) + pickle.dumps(kwargs, protocol=4)
    ).hexdigest()

    dst_path = cache_folder() / f"{prefix}_{hash_key}"
    return dst_path.exists(), dst_path
