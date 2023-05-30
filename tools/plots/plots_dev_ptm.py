from plaster.tools.plots import plots
from plaster.tools.zplots.zplots import ZPlots


def plot_pr_breakout_peps_runs(job, peps_runs_df, filters, **kwargs):
    # TODO: see similar in plots_dev_mhc.py when plotting from df
    # This is only being called by PTM template at the moment I think.
    # Move to dev_ptm module

    """
    Single plot of PR curves for run_i+pep_i pairs given in peps_runs_df

    job: the JobResult that contains all of the RunResult objects indexed by run_i
    peps_runs_df: a df containing run_i,run_name,pep_i,ptm per row whose PR should be plotted
    """

    z = ZPlots.zplot_singleton
    z.color_reset()
    n_peps = len(peps_runs_df.pep_i.unique())
    title = kwargs.pop(
        "f_title", f"PR curves, {n_peps} peptides, best runs ({filters.classifier})"
    )
    with z(
        _merge=True,
        f_y_axis_label="precision",
        f_x_axis_label="read recall",
        f_title=title,
        _legend="bottom_right",
        **kwargs,
    ):
        # first pass to collect data so it can be sorted and affect legend order
        pr_data = []
        for (run_i, pep_i), row in peps_runs_df.groupby(["run_i", "pep_i"]):
            run_name = row.run_name.iloc[0]
            ptms = ";".join(list(row.ptm.astype(str)))
            ptms = f"({ptms})" if ptms else ""
            name = f'r{run_i}p{pep_i}{ptms}{"_".join(run_name.split("_")[:-1])}'
            # tuple[3] is to allow sort on PTM,reverse-precision
            pr_data += [
                (run_i, pep_i, name, f"{row.ptm.iloc[0]}prec{1-row.prec.iloc[0]:.4f}")
            ]

        pr_data.sort(key=lambda tup: tup[3])

        # second pass to plot data
        for (run_i, pep_i, name, first_ptm) in pr_data:
            plots.plot_pr_breakout(
                job.runs[run_i],
                pep_iz=[pep_i],
                color=z.next(),
                legend_label=name,
                _noise=0.005,
            )
