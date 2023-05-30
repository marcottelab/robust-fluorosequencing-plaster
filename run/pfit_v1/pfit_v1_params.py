from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dataclasses_json import DataClassJsonMixin


@dataclass
class ParamInfo(DataClassJsonMixin):
    name: str
    initial_value: Optional[float] = None
    is_fixed: bool = False
    bound_lower: float = 0.0
    bound_upper: float = 1.0


@dataclass
class ParameterEstimationParams(DataClassJsonMixin):
    exp_name: str
    dyeseqs: list[str]
    n_ch: int
    n_pres: int
    n_edmans: int
    n_mocks: int
    marker_labels: list[str]
    true_dyetracks: list[str]
    true_dyetracks_count: list[int]
    plaster_x0: dict[str, float]

    x0: Optional[list[float]] = None
    defaults: Optional[dict[str, ParamInfo]] = None
    n_reps: int = 10
    default_n_samples: int = 1_000_000

    # I have to make this optional because older saved versions of this don't have it.
    n_span_labels_list: Optional[list[int]] = None

    # This is deprecated, but I have to include it because older saved versions try to load it.
    n_span_labels: int = -1


@dataclass
class PlasterPaths(DataClassJsonMixin):
    job: Path
    vfs_job: Path
    classify_job: Path
    sim_job: Path
    sigproc_job: Path
    filter_params_yaml_path: Path


def import_plaster(paths: PlasterPaths) -> ParameterEstimationParams:
    from plaster.genv2.gen_utils import resolve_job_folder

    job = resolve_job_folder(paths.job)
    vfs_path = resolve_job_folder(paths.vfs_job)
    classify_path = resolve_job_folder(paths.classify_job)

    job.mkdir(parents=True, exist_ok=True)

    rf_path = classify_path / "rf_classify"

    import plaster.genv2.generators.vfs as vfsgen

    sim_path = None
    sim_result = None
    filter_df = None

    if paths.sim_job is not None:
        sim_result = vfsgen.SimV3FlyteResult.load_from_disk(
            vfs_path / "sim"
        ).load_result()

        sim_job_path = resolve_job_folder(paths.sim_job)
        sim_path = sim_job_path / "sim"
        prep_path = sim_job_path / "prep"
    else:
        prep_path = vfs_path / "prep"

        sim_path = resolve_job_folder(paths.vfs_job)
        sim_path = vfs_path / "sim"
        sim_result = vfsgen.SimV3FlyteResult.load_from_disk(sim_path).load_result()

        from plaster.run.ims_import.ims_import_result import (
            ImsImportFlyteResult,
            ImsImportResult,
        )
        from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2FlyteResult

        experiment_path = paths.sigproc_job

        experiment_path = resolve_job_folder(experiment_path)
        sigproc_path = experiment_path / "sigproc_v2"
        ims_import_path = experiment_path / "ims_import"
        sigproc_result = SigprocV2FlyteResult.load_from_disk(sigproc_path).load_result()
        ims_import_result = ImsImportFlyteResult.load_from_disk(
            ims_import_path
        ).load_result()

        from plaster.reports.helpers.params import ReportParams
        from plaster.reports.helpers.report_params import FilterParams

        report_params = ReportParams.defaults(sigproc_result.n_channels)
        report_params.load(experiment_path)

        filter_params = FilterParams.load_from_path(
            resolve_job_folder(paths.filter_params_yaml_path)
        )
        # filter_params = FilterParams.from_report_params(report_params)

        if not filter_params.check_all_present():
            raise ValueError("Some filtering parameters have not been specified!")

        import plaster.reports.helpers.report_params_filtering as pf

        filter_df = pf.get_classify_default_filtering_df(
            sigproc_result, ims_import_result, filter_params, all_cycles=True
        )

    rf_classify_result = vfsgen.RFV2FlyteResult.load_from_disk(rf_path).load_result()
    prep_result = vfsgen.PrepV2FlyteResult.load_from_disk(prep_path).load_result()

    n_channels = sim_result.n_channels
    n_pres = sim_result.params.seq_params.n_pres
    n_edmans = sim_result.params.seq_params.n_edmans
    n_mocks = sim_result.params.seq_params.n_mocks
    n_span_labels_list = sim_result.params.n_span_labels_list

    from collections import Counter

    import numpy as np
    import pandas as pd

    def dytmat2txt(dytmat: np.array) -> list[str]:
        """Convert a SimV3Result dytmat array to a list of dyetrack strings."""
        return ["".join(map(str, dytmat[k, :])) for k in range(1, dytmat.shape[0])]

    score_threshold = 0.0
    # build classification result
    rfc_df = pd.DataFrame.from_dict(
        {
            x: rf_classify_result.__getattribute__(x).get()
            for x in [
                "pred_pep_iz",
                "runnerup_pep_iz",
                "scores",
                "runnerup_scores",
            ]
        },
        orient="columns",
    )

    rfc_filter = rfc_df
    if filter_df is not None:
        rfc_filter = rfc_filter.loc[filter_df.pass_all, :]

    scored_rfc_filter = rfc_filter.loc[
        rfc_filter.loc[:, "scores"] >= score_threshold, :
    ]
    score_counts_filter = pd.DataFrame.from_dict(
        Counter(scored_rfc_filter.pred_pep_iz), orient="index", columns=["counts"]
    ).sort_index()

    # place dyt_labels on the score_counts_filter
    classifier_dyt_labels = pd.Series(
        dytmat2txt(sim_result.train_dytmat),
        name="dyt_labels",
        index=range(1, sim_result.train_dytmat.shape[0]),
    )
    score_counts_filter = score_counts_filter.join(classifier_dyt_labels)

    a0 = score_counts_filter.sort_values(by=["dyt_labels", "counts"])[
        ["dyt_labels", "counts"]
    ].values.tolist()

    a = [[x[0], x[1]] for x in a0]
    exp_dyetracks = [x[0] for x in a]
    exp_dyetracks_count = [x[1] for x in a]

    exp_name = "unknown"

    # TODO Which of these two is best?
    marker_labels = [x.aa for x in sim_result.params.markers]
    # marker_labels = [x.aa for x in sim_result.params.labels]

    # TODO This probably is not correct if there are proteases. In that
    # case, one may need to build a list of peptide sequences using
    # the pro and pep lists.
    proteins = [x.sequence for x in prep_result.params.proteins]
    dyeseqs = [
        "".join([x if x in marker_labels else "." for x in seq]) for seq in proteins
    ]

    true_dyetracks, true_dyetracks_count = exp_dyetracks, exp_dyetracks_count

    from plaster.run.sim_v3.sim_v3_params import Marker, SeqParams

    plaster_x0 = {
        "p_initial_block": SeqParams.p_initial_block,
        "p_cyclic_block": SeqParams.p_cyclic_block,
        "p_detach": SeqParams.p_detach,
        "p_edman_failure": SeqParams.p_edman_failure,
        **{
            k: v
            for i, x in enumerate(marker_labels)
            for (k, v) in zip(
                [f"p_bleach_ch{i}", f"p_dud_ch{i}"], [Marker.p_bleach, Marker.p_dud]
            )
        },
    }

    rv = ParameterEstimationParams(
        exp_name=exp_name,
        dyeseqs=dyeseqs,
        n_ch=n_channels,
        n_pres=n_pres,
        n_edmans=n_edmans,
        n_mocks=n_mocks,
        n_span_labels_list=n_span_labels_list,
        marker_labels=marker_labels,
        true_dyetracks=true_dyetracks,
        true_dyetracks_count=true_dyetracks_count,
        plaster_x0=plaster_x0,
    )

    return rv
