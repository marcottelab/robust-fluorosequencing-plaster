from plumbum import local

from plaster.tools.utils import utils


def get_user():
    user = local.env.get("RUN_USER")
    if user is None or user == "":
        raise Exception("User not found in $USER")
    return user


def validate_job_folder(job_folder, search_if_not_present=False):
    """
    job_folder can be Python symbols.
    returns plumbum local path
    """
    basename = local.path(job_folder).name
    if not utils.is_symbol(basename):
        raise ValueError(
            f"job name must be a lower-case Python symbol (ie start with [a-z_] followed by [a-z0-9_] found '{basename}'"
        )

    p = local.path(job_folder)
    if p.exists():
        return p

    if search_if_not_present:
        _p = local.path(local.env["JOBS_FOLDER"]) / job_folder
        if _p.exists():
            return _p

    return p
