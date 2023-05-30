import pandas as pd
from IPython.display import display  # for display of DataFrames

from plaster.tools.ipynb_helpers.displays import hd
from plaster.tools.plots import plots, plots_dev
from plaster.tools.utils.utils import json_print, munch_abbreviation_string
from plaster.tools.zplots.zplots import ZPlots


def plot_best_runs_peptide_yield(best_pr, run_info, filters, **kwargs):
    """
    For each run, indicate how many peptides it was the 'best run' for based on
    filter criteria.
    """
    total_peps = len(best_pr.pep_i.unique())
    fracs_by_run = run_info.pep_counts / total_peps
    classifier_name = filters.get("classifier", "").upper()
    title = f'"Best PR" peptide-yield for runs that produced a best peptide-pr ({classifier_name})'
    y_label = "fraction of total peptides"
    x_range = run_info.run_labels
    z = ZPlots.zplot_singleton
    z.cols(
        fracs_by_run,
        x=x_range,
        f_x_range=x_range,
        _x_axis_label_orientation=1.2,
        f_title=title,
        f_y_axis_label=y_label,
        f_x_axis_label="run name",
        _label=run_info.pep_counts,
        _size_x=1000,
    )


def plot_best_runs_peptide_observability(
    job, best_pr, run_info, all_runs_pr, filters, **kwargs
):
    """
    peptide observability-vs-precision considering the best runs for each peptide
    as if they are one big super-run -- how well can we see peptides vs precision
    if the best run (as defined by filters) is chosen for each peptide?
    """
    z = ZPlots.zplot_singleton
    z.color_reset()
    classifier_name = filters.get("classifier", "").upper()
    with z(
        _merge=True,
        f_title=f"Peptide-Classes Precision/Recall (best {filters.plot_n_runs} + combined-best runs) ({classifier_name})",
        f_y_axis_label="precision",
        f_x_axis_label="peptide-classes recall",
    ):
        best_runs_full_pr = []
        for i, (run_i, n_pep, peps) in enumerate(
            zip(run_info.run_iz, run_info.pep_counts, run_info.peps)
        ):
            best_runs_full_pr += [
                all_runs_pr[
                    (all_runs_pr.run_i == run_i) & (all_runs_pr.pep_i.isin(peps))
                ]
            ]
            run = job.runs[run_i]
            if i < filters.plot_n_runs:
                label = f"{plots_dev._run_labels(run.run_name)} ({n_pep})"
                plots.plot_peptide_observability_vs_precision(
                    run,
                    pep_iz=filters.peptide_subset,
                    color=z.next(),
                    pr_axes=True,
                    _label=label,
                    legend_label=label,
                    _legend="top_right",
                    _range=(0, 1.05, 0, 1.05),
                )
        best_full_pr = pd.concat(best_runs_full_pr)
        plots._plot_peptide_observability_vs_precision(
            best_full_pr,
            color=z.next(),
            _label="combined",
            legend_label="combined (filters)",
            **kwargs,
        )


def plot_pr_scatter_peps_runs(peps_runs_df, run_info, **kwargs):
    """
    Single plot of best PR for run_i+pep_i pairs given in peps_runs_df

    peps_runs_df: a df containing run_i,run_name,pep_i,prec, and recall
    run_info: a Munch containing run_iz,run_labels,pep_counts,and peps per run, sorted

    """
    df = peps_runs_df.copy()
    df["label"] = df.apply(
        lambda x: f"{x.pep_i:03d} {x.seqstr} {x.flustr} ({x.flu_count})", axis=1
    )

    z = ZPlots.zplot_singleton
    n_peps = len(peps_runs_df.pep_i.unique())
    title = kwargs.get(
        "f_title", f"{n_peps} peptides, best precision for recall-filter"
    )
    z.color_reset()
    with z(
        _merge=True,
        f_y_axis_label="precision",
        f_x_axis_label="read recall",
        f_title=title,
        _legend="bottom_right",
        _range=(0, 1.05, 0, 1.05),
    ):
        groups = df.groupby("run_i")
        for run_i, run_label, pep_count in zip(
            run_info.run_iz, run_info.run_labels, run_info.pep_counts
        ):
            try:
                group = groups.get_group(run_i)
            except KeyError:
                continue  # the run has no entries in the DF, that's ok.
            legend = f"{run_label} ({pep_count})"
            z.scat(
                source=group,
                y="prec",
                x="recall",
                _label="label",
                fill_alpha=0.8,
                color=z.next(),
                legend_label=legend,
            )


def plot_best_runs_scatter(best_pr, run_info, filters, **kwargs):
    runs_to_plot = run_info.run_iz[: filters.plot_n_runs]
    n_peps = len(best_pr.pep_i.unique())
    plotted_pr = best_pr[best_pr.run_i.isin(runs_to_plot)]
    n_plotted_peps = len(plotted_pr.pep_i.unique())
    title = f"{n_plotted_peps} of {n_peps} peptides, best precision for min_recall={filters.min_recall} {filters.classifier}"
    plot_pr_scatter_peps_runs(plotted_pr, run_info, f_title=title, **kwargs)


def plot_best_runs_pr(best_pr, all_pr, run_info, filters, **kwargs):
    df = best_pr.sort_values(by=["prec", "recall"], ascending=[False, False])[
        : filters.plot_n_peps
    ]
    z = ZPlots.zplot_singleton
    z.color_reset()
    title = f"PR curves, best {len(df.pep_i.unique())} peptides, best runs. {filters.classifier} "
    run_i_to_info = {
        run_i: (run_label, z.next())
        for run_i, run_label in zip(run_info.run_iz, run_info.run_labels)
    }
    with z(
        f_title=title,
        _merge=True,
        _legend="bottom_right",
        f_y_axis_label="precision",
        f_x_axis_label="read recall",
    ):
        for i, row in df.iterrows():
            run_i = row.run_i
            pep_i = row.pep_i
            legend_label = f"{run_i_to_info[run_i][0]} p{pep_i}"
            line_label = f"{row.pep_i:03d} {row.seqstr} {row.flustr} ({row.flu_count})"
            color = run_i_to_info[run_i][1]
            prdf = all_pr[(all_pr.run_i == run_i) & (all_pr.pep_i == pep_i)]
            prsa = (prdf.prec.values, prdf.recall.values, prdf.score.values, None)
            plots.plot_pr_curve(
                prsa,
                color=color,
                legend_label=legend_label,
                _label=line_label,
                **kwargs,
            )


def show_best_runs_df(best_pr, filters, save_csv=True):
    hd(
        "h3",
        f"Top 50 precisions at min_recall={filters.min_recall} {filters.classifier}",
    )
    if save_csv:
        csv_filename = f"./report_best_pr__{munch_abbreviation_string(filters)}.csv"
        best_pr.to_csv(csv_filename, index=False, float_format="%g")
        print(f"(All {len(best_pr)} rows exported to {csv_filename})")
    pd.set_option("display.max_columns", None)
    display(best_pr.head(50))


def print_titles(filters):
    hd("h1", "Best runs per peptide")
    hd("h3", "Filters")
    json_print(filters)
    print()
