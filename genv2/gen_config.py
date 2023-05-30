import dataclasses
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from dataclasses_json import DataClassJsonMixin
from flytekit.core.workflow import WorkflowBase

from plaster.genv2.legacy_utils import report_builder
from plaster.tools.utils import utils

log = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    valid: bool
    message: Optional[str] = None


def gen_field(
    default: Any = dataclasses.MISSING,
    default_factory: Type = dataclasses.MISSING,
    help: Optional[str] = None,
    validator: Callable[..., ValidationResult] = None,
    download: bool = False,
    path: bool = False,
    hidden: bool = False,
    metadata_config: Dict[str, dict] = None,
) -> field:
    """
    This generates a helpful warning message when no help is provided, such as:
        No help provided: ch_aln: Optional[List[float]] = gen_field(default=False) in class SigprocV2Config(BaseGenConfig):

    params:
        download: bool - If True, the value of the field will be downloaded and converted to a local path before being handed to the generator.
        path: bool - If True, the value of the field is expected to be a local path.
        hidden: bool - Don't show help for the field.
    """
    if help is None:
        stack = inspect.stack()
        line = stack[1].code_context[0].strip()
        container = stack[2].code_context[0].strip()
        log.warning(f"No help provided: {line} in {container}")

    metadata = {}
    if help:
        metadata["help"] = help

    if validator:
        metadata["validator"] = validator

    if download:
        metadata["download"] = download

    if path:
        metadata["path"] = path

    if metadata_config:
        metadata.update(metadata_config)

    return field(
        default=default,
        default_factory=default_factory,
        metadata=metadata,
    )


@dataclass
class BaseGenConfig(DataClassJsonMixin):
    """
    Note: since this will be subclassed by other dataclasses, all fields here must be non-optional (no default value)
    """

    type: str = gen_field(help="This value is used to determine which generator to use")
    job: str = gen_field(help="Path to the job folder", path=True)

    # I want to be able to track ownership of generated jobs, and I think I can do this
    # by causing all configs to have an owner attribute, which can be automatically
    # populated in the POST to ControlPanel (for either GenV2 or Flyte) where
    # request.user_data.email is available.  For Flyte jobs in particular, I need to
    # push this information into the config sent to Flyte, because I'll need that
    # owner information in a @task function which writes the job_manifest.yaml to
    # the jobs_folder.  In the case of GenV2 jobs, that EFS job_manifest file is written
    # immediately as part of job launch, but for Flyte, it is written as a special sidecar
    # task at the end of the workflow. Until this is better understood, see the hack in
    # gen_utils.py that is part of this PR.
    #
    # owner: str = gen_field(help="Who generated this job")

    def validate(self):

        # TODO(b7r6): do something more consistent across `genv2` and `JRI` once consensus emerges
        if self.job.startswith("/jobs_folder") or self.job.startswith("jobs_folder"):
            return ValidationResult(
                valid=False,
                message=f"To refer to `jobs_folder` use the format `//jobs_folder/...`, found: {self.job}",
            )

        # Parse the path
        try:
            path = Path(self.job)
        except TypeError as e:
            return ValidationResult(False, str(e))

        # Ensure the job name is a valid python symbol
        if not utils.is_symbol(path.name):
            return ValidationResult(
                valid=False,
                message=f"job must be a lower-case Python symbol (ie start with [a-z_] followed by [a-z0-9_] found '{self.job}'",
            )

        return ValidationResult(True)


@dataclass
class BaseGenConfigDefaultArgs(DataClassJsonMixin):
    force: bool = gen_field(default=False, help="Force deletion of existing job")
    report_params: Optional[Dict] = gen_field(
        default=None,
        help="Configuration related to reports (see reports for specific fields)",
    )


@dataclass
class GenerateResult:
    # The list of runs to generate
    runs: List[dict]
    # The path to the job folder (this is default None because it will get set after the generator is run)
    job_folder: Optional[Path] = None
    # The input config (Default none for same reason as job_folder)
    config: Type[BaseGenConfig] = None
    # A list of report builder reports to generate
    reports: Dict[str, report_builder.ReportBuilder] = field(default_factory=dict)
    # A list of static report files to copy over
    static_reports: List[str] = field(default_factory=list)

    def validate(self) -> ValidationResult:
        # check for no duplicate run names
        if len(self.runs) != len(set(run["run_name"] for run in self.runs)):
            return ValidationResult(False, "Duplicate run names")

        return ValidationResult(True)


GeneratorFunc = Callable[[BaseGenConfig], GenerateResult]


@dataclass
class Generator:
    config: Type[BaseGenConfig]
    generate: GeneratorFunc
    workflow: Optional[WorkflowBase] = None
