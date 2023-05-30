"""Generate and run pipeline tasks programmatically."""
import pathlib
import tempfile
import typing
from enum import Enum

import nbconvert.exporters
import nbconvert.writers
import papermill as pm
from traitlets.config import Config

import plaster.run.run as run
import plaster.tools.utils.utils as utils


class Extensions(Enum):
    ipynb = ".ipynb"
    html = ".html"


def _export_html(infile: typing.IO[bytes], outfile_path: str) -> None:
    """Export notebook to html format and write it to output file.

    Arguments:
        infile: the executed notebook to convert
        outfile_path: HTML file path
    """
    (output, resources) = nbconvert.exporters.HTMLExporter(
        config=Config(
            {
                "ExecutePreprocessor": {"enabled": True, "timeout": 28800},
                "TemplateExporter": {
                    "exclude_output_prompt": True,
                    "exclude_input": True,
                    "exclude_input_prompt": True,
                },
            }
        ),
        template_name="lab",
    ).from_file(infile)

    # writer add .html suffix
    nbconvert.writers.FilesWriter().write(output, resources, notebook_name=outfile_path)


def run_ipynb(nb_path: pathlib.Path) -> None:
    """Execute specified Jupyter notebook and convert it to HTML."""
    with tempfile.NamedTemporaryFile() as tf:
        # executing notebook to temp file
        pm.execute_notebook(
            input_path=nb_path,
            kernel_name="python3",
            output_path=tf.name,
        )
        _export_html(tf, str(nb_path.with_suffix("")))


def run_job(
    job_folder: str,
    force: bool = False,
    no_progress: bool = True,
    run_reports: bool = True,
) -> typing.Dict[pathlib.PosixPath, int]:
    job_folder = pathlib.Path(job_folder)

    # Find all the dirs with plaster_run.yaml files. They might be in run subfolders
    run_dirs = [x.parent for x in job_folder.rglob("plaster_run.yaml")]
    if not run_dirs:
        raise OSError(
            "Plaster: Nothing to do because no run_dirs have plaster_run.yaml files"
        )
    status = {}
    for run_dir in run_dirs:
        status[run_dir] = (
            run.RunExecutor(run_dir)
            .load()
            .execute(force=force, no_progress=no_progress)
        )

    report_paths = job_folder.rglob("*" + Extensions.ipynb.value)

    if run_reports:
        for report_src_path in report_paths:
            report_dst_path = report_src_path.with_suffix(Extensions.html.value)
            if report_src_path.exists() and (
                force or utils.build_utils.out_of_date(report_src_path, report_dst_path)
            ):
                print(f"Running report {report_src_path}")
                run_ipynb(report_src_path)

    return status
