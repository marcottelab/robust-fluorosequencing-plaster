import pathlib
import shutil
import time
from typing import List, Mapping

import flytekit
import yaml

from plaster import root
from plaster.reports.helpers.params import ReportParams
from plaster.tools.flyte import task_configs


def resolve_job_folder(job_folder: str) -> pathlib.Path:
    if job_folder.startswith("//jobs_folder/"):
        job_folder = job_folder.replace("//jobs_folder/", "")
        job_path = pathlib.Path(task_configs.get_job_folder_mount_path()) / job_folder
    else:
        job_path = pathlib.Path(job_folder)

    return job_path


def write_job_folder(
    job_folder: str,
    config_dict: Mapping[str, object],
    clean: bool = True,
    manifest_name="job_manifest.yaml",
) -> pathlib.Path:
    job_path = resolve_job_folder(job_folder)

    if clean and job_path.exists():
        shutil.rmtree(job_path)

    job_path.mkdir(parents=True, exist_ok=(not clean))

    # We would like to populate the "who" field below with something useful,
    # but it's unclear the best way to do so at present.  This fn is called
    # from Flyte @task fns, which get passed a config object.  Prob we can
    # add an 'owner' field to BaseGenConfig and set that in the POST to
    # controlpanel based on request.user_data.email, but I'm very tired
    # and want people to be able to start using flyte workflows tomorrow.
    # An advantage of this little hack is that it works even for jobs
    # not launched via the ControlPanel - so devs executing from notebooks
    # should see their jobs in Indexer searchable by their name.
    #
    # So... just look at the jobs_folder path and infer ownership from that.
    # I'll prepend a slash as a reminder that this is an inferred ownership
    # based on the path.  tfb 09 Nov 22
    import re

    try:
        owner = "/" + re.search(r"jobs_folder/(.+?)/", str(job_path))[1]
    except:
        owner = "/Unknown"

    # Write job manifest
    # Note that these fields match the legacy job manifest format
    job_manifest = {
        "id": flytekit.current_context().execution_id.name,
        "localtime": time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime()),
        "cmdline_args": [],
        "config": config_dict,
        "who": owner,
    }

    with open(job_path / manifest_name, "w") as f:
        yaml.safe_dump(job_manifest, f)

    return job_path


def write_default_sigproc_report_params(job_path: pathlib.Path, n_channels: int):
    """
    Writes a commented-out version of the _reports_params.yaml file into the passed folder,
    using default values from ReportParams.  The values that are required by FilterParams
    are sorted to the top of the file, so that humans charged with editing and blessing params
    for sigproc data-filtering can more easily know which parameters are required.
    """
    report_dir = job_path / "_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    default_params = ReportParams.defaults(n_channels)
    with open(report_dir / "_report_params.yaml", "w") as f:
        f.write(default_params.get_yaml_for_param_sections(commented_out=True))


def write_reports(job_path: pathlib.Path, reports: List[str], exist_ok=False):
    """
    Reports should be a list of strings, where each string is a path relative to plaster/reports

    Example:
    write_reports(job_path, [
        "sigproc_fields.ipynb",
        "sigproc_radiometry.ipynb",
        "sigproc_fret.ipynb",
        "sigproc_chcolo.ipynb",
    ])
    """
    report_dir = job_path / "_reports"

    if report_dir.exists():
        shutil.rmtree(report_dir)

    report_dir.mkdir(parents=True, exist_ok=exist_ok)

    report_source_dir = root.location() / "reports"

    for report in reports:
        report_path = pathlib.Path(report)
        source_file = report_source_dir / report_path
        dest_file = report_dir / report_path.name
        shutil.copy(source_file, dest_file)
