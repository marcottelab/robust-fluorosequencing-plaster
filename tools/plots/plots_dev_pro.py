from plaster.tools.plots import plots
from plaster.tools.zplots.zplots import ZPlots


# This is taken directly from the same-named fn in plots_dev_mhc.
# The only difference here is titles and labeling for proteins and not peptides.
#
def plot_best_runs_pr(best_pr, all_pr, run_info, filters, **kwargs):
    df = best_pr.sort_values(by=["prec", "recall"], ascending=[False, False])
    z = ZPlots.zplot_singleton
    max_traces = filters.get("max_traces", plots.MAX_BOKEH_PLOT_TRACES)
    title = f"{filters.classifier} PR curves, protein identification ({len(df.pro_i.unique())} proteins), best runs."
    with z(
        f_title=title,
        _merge=True,
        _legend="bottom_right",
        f_y_axis_label="precision",
        f_x_axis_label="read recall",
    ):
        trace_count = 0
        color_by_run = len(run_info.run_iz) > 1
        groups = df.groupby("run_i")
        for run_i, run_label in zip(run_info.run_iz, run_info.run_labels):
            if trace_count >= max_traces:
                break

            group = groups.get_group(run_i)
            if color_by_run:
                color = z.next()

            for i, row in group.iterrows():
                pep_i = row.pep_i
                pro_id = row.pro_id

                if not color_by_run:
                    color = z.next()

                legend_label = f"{run_label} {pro_id} {pep_i}"
                line_label = (
                    f"{run_label} {pro_id} pep{row.pep_i:03d} {row.seqstr} {row.flustr}"
                )
                prdf = all_pr[(all_pr.run_i == run_i) & (all_pr.pep_i == pep_i)]
                prsa = (prdf.prec.values, prdf.recall.values, prdf.score.values, None)
                plots.plot_pr_curve(
                    prsa,
                    color=color,
                    legend_label=legend_label,
                    _label=line_label,
                    **kwargs,
                )

                trace_count += 1
                if trace_count >= max_traces:
                    break
