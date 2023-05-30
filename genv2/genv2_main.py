import dataclasses
import json
import logging
import shutil
import sys
import tempfile
import time
import typing
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Type, Union

import marshmallow
import structlog
import yaml
from munch import Munch
from plumbum import cli, local

from plaster.genv2.gen_config import (
    BaseGenConfig,
    BaseGenConfigDefaultArgs,
    GenerateResult,
    Generator,
    ValidationResult,
)
from plaster.genv2.generators import (
    classify,
    pfit,
    sigproc_v2,
    sigproc_v2_calib,
    sim,
    survey,
    vfs,
    whatprot,
)
from plaster.genv2.generators.utils import Classifier

# from plaster.genv2.legacy_utils import report_builder
from plaster.tools.utils import tmp
from plaster.tools.zlog import zlog

logger = structlog.get_logger()


GENERATORS = {
    "classify": classify.generator,
    "classify_big": classify.generator_big,
    "sigproc_v2_calib": sigproc_v2_calib.generator,
    "sigproc_v2": sigproc_v2.generator,
    "sim": sim.generator,
    "sim_big": sim.generator_big,
    "sim_xl": sim.generator_xl,
    "survey": survey.generator,
    "vfs": vfs.generator,
    "vfs_big": vfs.generator_big,
    "whatprot_classify": whatprot.generator_classify,
    "whatprot_classify_xl": whatprot.generator_classify_xl,
    "whatprot_fit": whatprot.generator_fit,
    "param_fit": pfit.generator_param_fit,
    "param_fit_easy": pfit.generator_param_fit_easy,
    "param_fit_vfs": pfit.generator_param_fit_vfs,
    "parameter_estimation_fit": pfit.generator_fit,
    "parameter_estimation_precheck": pfit.generator_precheck,
}


@dataclass
class ParseResult:
    data: Optional[Type[BaseGenConfig]] = None
    generator: Optional[Generator] = None
    error: Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.error is None


def validate(obj):
    """
    Validate an object where it or its child elements have validate methods.
    """
    if isinstance(obj, Classifier):
        return ValidationResult(True)

    if hasattr(obj, "validate"):
        validation = obj.validate()
        if not validation.valid:
            return validation

    if hasattr(obj, "__dict__"):
        for field in vars(obj).values():
            validation = validate(field)
            if not validation.valid:
                return validation

    return ValidationResult(True)


def parse(input: str) -> ParseResult:
    """
    Given a raw config string, returns a dataclass with the parsed data.

    This function should have no side effects.

    Returns a ParseResult object, which, if parsing fails, should have helpful errors messages intended for an end user.
    """

    # First parse the yaml
    try:
        data_dict = yaml.safe_load(input)
    except yaml.YAMLError as e:
        # TODO: This is probably a MarkedYAMLError, and thus we can extract line number info and give a very detailed error with the problem
        return ParseResult(error=e)

    if not data_dict or not isinstance(data_dict, dict):
        return ParseResult(error="config file is not a valid yaml dictionary")

    # Before we turn it into a dataclass, extract the type key so we can grab the correct generator config dataclass
    key = data_dict.get("type")
    if not key:
        return ParseResult(
            error="No gen type specified in config. Use 'type: <gen_type>'"
        )
    generator = GENERATORS.get(key)
    if not generator:
        return ParseResult(error=f"Unknown generator type: {key}")

    # This little hack lets us define default args for all generators
    @dataclass
    class ParsedConfig(
        BaseGenConfigDefaultArgs,
        generator.config,
    ):
        pass

    # Convert the dict into the dataclass
    try:
        # .schema generates a marshmallow schema from the dataclass,
        # which when used to parse the dict will correctly raise validation error for types
        data = ParsedConfig.schema().load(data_dict)
    except marshmallow.ValidationError as e:
        return ParseResult(error=e.messages)
    except Exception as e:
        # TODO: this could be done a bit better to give a more helpful error message
        return ParseResult(error=e)

    # Run custom generator specific validation logic
    validation = validate(data)
    if not validation.valid:
        return ParseResult(error=validation.message)

    return ParseResult(data=data, generator=generator)


# class GenInstance:
#     """
#     This class holds various information related to a single invocation of gen
#     """

#     def __init__(self):
#         self.uuid = uuid.uuid4().hex
#         self.tmp_dir = Path(tempfile.gettempdir()) / self.uuid
#         self.tmp_dir.mkdir(parents=True, exist_ok=True)


# def resolve_source(source, tmp_dir, symlink_to_cache=False):
#     if source.startswith("s3:"):
#         found_cache, cache_path = tmp.cache_path("plaster_s3", source)
#         if not found_cache:
#             logger.info("Syncing", source=source, cache_path=cache_path)

#             # creating logging instance to keep this interface the same
#             # while refactoring the rest of zlog
#             log = logging.getLogger(__name__)
#             local["aws"]["s3", "sync", source, cache_path] & zlog.ZlogFG(
#                 logger=log, drop_dup_lines=True, convert_to_cr=True
#             )

#         # A little bit of a hack to apply this to an url but it does what we want
#         source_folder_name = local.path(source).name

#         if symlink_to_cache:
#             local["ln"][
#                 "-s",
#                 cache_path,
#                 tmp_dir / source_folder_name,
#             ]()
#         else:
#             local["cp"][
#                 "-r",
#                 cache_path,
#                 tmp_dir / source_folder_name,
#             ]()

#         # Hardcoding relative path here. It will always be the same 3 levels: run/plaster_output/task
#         return "../../../_gen_sources/" + source_folder_name

#     return source


def rgetattr(obj, path):
    attrs = path.split(".")
    for attr in attrs:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return None
    return obj


def rsetattr(obj, path, value):
    attrs = path.split(".")
    last_attr = attrs.pop()
    for attr in attrs:
        obj = getattr(obj, attr)
    setattr(obj, last_attr, value)


def find_fields_with_metadata(obj, metadata_key, *, current_path=""):
    """
    Find all fields in a dataclass that have the given metadata key.
    """

    def get_full_path(field_name):
        if current_path:
            return f"{current_path}.{field_name}"
        else:
            return field_name

    fields = []
    for field in dataclasses.fields(obj):
        if metadata_key in field.metadata:
            fields.append(get_full_path(field.name))
        if dataclasses.is_dataclass(field.type):
            fields.extend(
                find_fields_with_metadata(
                    getattr(obj, field.name),
                    metadata_key,
                    current_path=get_full_path(field.name),
                )
            )
    return fields


def apply_substitution(config, metadata_key, sub_fn):
    """
    Given a config, replace every field (in place) with metadata_key with the result of sub_fn(field_value)
    """
    fields = find_fields_with_metadata(config, metadata_key)

    for field in fields:
        original = rgetattr(config, field)
        substituted = sub_fn(original)
        rsetattr(config, field, substituted)


# def resolve_source_paths(config, tmp_dir):
#     """
#     This modifies the given config in place.
#     """
#     sources = find_fields_with_metadata(config, "download")

#     for source in sources:
#         source_path = rgetattr(config, source)
#         resolved_path = resolve_source(source_path, tmp_dir)
#         rsetattr(config, source, resolved_path)


# def write_run(job_folder, run_descs):
#     """
#     Convert the munch run_descs into folders
#     """

#     found_run_names = {}

#     for i, run in enumerate(run_descs):
#         # FIND run_name
#         run_name = run.get("run_name")

#         if run_name in found_run_names:
#             raise Exception(f"More than one run with name {run_name} found")
#         found_run_names[run_name] = True

#         # SETUP _erisyon block
#         if "_erisyon" not in run:
#             run._erisyon = Munch()
#         run._erisyon.run_i = i
#         run._erisyon.run_i_of = len(run_descs)
#         run._erisyon.run_name = run_name

#         # Keep the run_name out
#         run.pop("run_name", None)
#         folder = job_folder / run_name
#         folder.mkdir()
#         RunExecutor(folder, tasks=run).save()

#         logger.info(f"Wrote run", folder=folder)


# def write_report_builder_reports(
#     job_folder, reports: Dict[str, report_builder.ReportBuilder]
# ):
#     for report_name, report_builder in reports.items():
#         report = report_builder.report_assemble()
#         if report is not None:
#             # TODO: should these go in the _reports folder?
#             (job_folder / f"{report_name}.ipynb").write_text(json.dumps(report))


# def write_static_reports(job_folder, static_reports):
#     for report_name in static_reports:
#         if report_name is not None:
#             report_name = f"{report_name}.ipynb"
#             src = Path(__file__).resolve().parent.parent / "reports" / report_name
#             dst_folder = job_folder / "_reports"
#             dst = dst_folder / report_name
#             shutil.copy(src, dst)


# def write_generator_result(result: GenerateResult, gen_instance: GenInstance):
#     result.job_folder.mkdir(parents=True, exist_ok=True)

#     write_run(result.job_folder, result.runs)

#     (result.job_folder / "reports_archive").mkdir()
#     (result.job_folder / "_reports").mkdir()
#     write_report_builder_reports(result.job_folder, result.reports)
#     write_static_reports(result.job_folder, result.static_reports)

#     (result.job_folder / "job_manifest.yaml").write_text(
#         yaml.safe_dump(
#             dict(
#                 uuid=gen_instance.uuid,
#                 localtime=time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime()),
#                 # Note: it seems localtime inside our container is UTC
#                 who=local.env.get("RUN_USER", "Unknown"),
#                 cmdline_args=sys.argv,
#                 config=result.config.to_dict(),
#             )
#         )
#     )


# def run_gen(config_str: str, force: bool):
#     result = parse(config_str)

#     # Check that the config parsed correctly
#     if not result.valid:
#         logger.error("Failed to parse input", error=repr(result.error))
#         return 1

#     # Create an object that represents this invocation of gen
#     gen_instance = GenInstance()

#     # Resolve the source paths declared by the generator
#     resolve_source_paths(result.data, gen_instance.tmp_dir)

#     # Generate the job spec from the config
#     generate_result = result.generator.generate(result.data)
#     generate_result.job_folder = Path(result.data.job)
#     generate_result.config = result.data

#     # Validate the result of the generator
#     generator_validation = generate_result.validate()
#     if not generator_validation.valid:
#         logger.error("generator output", output=generator_validation.message)
#         return 1

#     # Blow away the old job folder if necessary
#     # TODO: maybe this should go in write_generator_result?
#     if force or result.data.force:
#         if generate_result.job_folder.exists():
#             shutil.rmtree(generate_result.job_folder)

#     # Write out the job folder
#     write_generator_result(generate_result, gen_instance)

#     # Write the raw config to the job folder
#     (generate_result.job_folder / "gen.yaml").write_text(config_str)

#     # Copy over the downloaded source files
#     shutil.move(gen_instance.tmp_dir, generate_result.job_folder / "_gen_sources")

#     return 0


def generate_help_text(gen_type: str):
    def comment(text):
        lines = text.strip().split("\n")
        return "\n".join([f"# {l}" for l in lines])

    def indent(text, n=2):
        lines = text.split("\n")
        return "\n".join([" " * n + l for l in lines])

    def is_optional(field):
        return typing.get_origin(field) is Union and type(None) in typing.get_args(
            field
        )

    def get_optional_type(type):
        """
        If a given type is Optinal[T], return T, otherwise return None
        """
        if is_optional(type):
            return type.__args__[0]
        else:
            return None

    def get_type_name(type):
        if hasattr(type, "__name__"):
            return type.__name__
        return repr(type)

    def generate_help_for_dataclass(cls):
        # This little hack is similar to what's taking place in parse, which gets us a dataclass with our global, optional fields
        @dataclass
        class DataclassWithGlobalOptionalFields(
            BaseGenConfigDefaultArgs,
            cls,
        ):
            pass

        help = ""

        fields = dataclasses.fields(DataclassWithGlobalOptionalFields)

        # Loop over all fields
        for i, field in enumerate(fields):
            # Don't show help for hidden fields
            if field.metadata.get("hidden"):
                continue

            field_help = field.metadata.get("help")
            if field_help:
                help += comment(field_help) + "\n"

            # This is sort of a hack such that the top of every generated help will show, for example, "type: sigproc_v2" instead
            # of showing "type: str"
            if field.name == "type" and issubclass(cls, BaseGenConfig):
                help += f"type: {gen_type}\n\n"
                continue

            # If the field type is Optional[T] extract T
            optional_type = get_optional_type(field.type)

            actual_type = optional_type if optional_type else field.type

            # If we're a dataclass, recurse
            if dataclasses.is_dataclass(actual_type):
                help += f"{field.name}: "
                if optional_type:
                    help += "# (optional)"
                help += "\n"
                help += indent(generate_help_for_dataclass(actual_type)) + "\n"
            else:
                help += f"{field.name}: "
                if optional_type:
                    help += f"{get_type_name(optional_type)} # (optional. default: {field.default})\n"
                else:
                    help += f"{get_type_name(field.type)}\n"

            # Add a new line to separate fields visually if we're not on the last field
            if i < len(fields) - 1:
                help += "\n"

        return help

    generator = GENERATORS.get(gen_type)

    return generate_help_for_dataclass(generator.config)


# class GenV2Application(cli.Application):
#     force = cli.Flag(["--force"], default=False, help="Force deletion of existing job")

#     def main(self, path: cli.ExistingFile = None):
#         if not self.nested_command:
#             input_str = path.read()

#             return run_gen(input_str, force=self.force)


# @GenV2Application.subcommand("generator-help")
# class GenerateHelp(cli.Application):
#     def main(self, gen_type: str):
#         print(generate_help_text(gen_type))


# if __name__ == "__main__":
#     GenV2Application.run()
