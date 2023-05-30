from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import Cycler, cycler

from plaster.genv2.generators.vfs import load_vfs_results
from plaster.run.prep_v2.prep_v2_result import PrepV2FlyteResult
from plaster.run.sim_v3.sim_v3_result import SimV3FlyteResult

colors = ["black", "brown", "orange", "turquoise", "steelblue", "indigo", "deeppink"]

color_cycler = cycler("color", colors)
marker_cycler = cycler("marker", ["o"])
color_marker_cycler = color_cycler * marker_cycler


def build_pr_info(
    path: str,
) -> Tuple[pd.DataFrame, PrepV2FlyteResult, SimV3FlyteResult]:
    _, prep_result, sim_result, _, rf_result = load_vfs_results(path)
    test_pr = {}
    score_cutoffs = np.linspace(0.1, 1.0, 10)
    for score_cutoff in score_cutoffs:
        this_rf_eval = rf_result.load_result().rf_sim_eval(
            score_cutoff=score_cutoff, try_runnerup=True
        )
        for pep_idx in this_rf_eval.index:
            if pep_idx not in test_pr:
                test_pr[pep_idx] = {"precision": [], "recall": [], "recall_corr": []}
            if (
                this_rf_eval.loc[pep_idx, "precision"] > 0.0
                and this_rf_eval.loc[pep_idx, "recall"] > 0.0
                and this_rf_eval.loc[pep_idx, "recall_corr"] > 0.0
            ):
                test_pr[pep_idx]["precision"].append(
                    this_rf_eval.loc[pep_idx, "precision"]
                )
                test_pr[pep_idx]["recall"].append(this_rf_eval.loc[pep_idx, "recall"])
                test_pr[pep_idx]["recall_corr"].append(
                    this_rf_eval.loc[pep_idx, "recall_corr"]
                )
    test_pr = pd.DataFrame.from_dict(test_pr, orient="index")
    return test_pr, prep_result, sim_result


def pr_plots(
    pr_info: pd.DataFrame,
    prep_result: PrepV2FlyteResult,
    sim_result: SimV3FlyteResult,
    cycler: Cycler = color_marker_cycler,
    figsize: Tuple[int, int] = (10, 8),
    fontsize: int = 12,
    recall_correction: bool = True,
):
    recall_column = "recall"
    if recall_correction:
        recall_column = "recall_corr"
    this_fret_int = sim_result.load_result().params.frets[0]
    this_edman_fail = sim_result.load_result().params.seq_params.p_edman_failure
    this_labels = sim_result.load_result().params.labels
    this_title = f"A:{this_labels[this_fret_int.acceptor].aa} D:{this_labels[this_fret_int.donor].aa} p_edman_failure:{(this_edman_fail):.2f}"
    quadp = sim_result.load_result().pros__peps__pepstrs__flus(
        prep_result.load_result()
    )
    cyc = cycler()
    plt.figure(figsize=figsize)
    for idx in pr_info.index:
        if len(pr_info.loc[idx, recall_column]) > 2:
            plt.plot(
                pr_info.loc[idx, recall_column],
                pr_info.loc[idx, "precision"],
                markerfacecolor="white",
                markersize=6,
                markeredgewidth=2,
                label=f"{idx}: {quadp.loc[idx, 'flustr']}",
                **next(cyc),
            )
    plt.legend(fontsize=fontsize, loc=(1.02, 0))
    plt.axis([0, 1.0, 0, 1.05])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel(recall_column, fontsize=fontsize)
    plt.ylabel("precision", fontsize=fontsize)
    plt.title(this_title, fontsize=fontsize)
    plt.tight_layout()
