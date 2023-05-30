import numpy as np
from IPython.core.display import display
from munch import Munch

from plaster.tools.ipynb_helpers.displays import hd
from plaster.tools.plots import plots
from plaster.tools.utils.utils import np_safe_divide
from plaster.tools.zplots.zplots import ZPlots

"""
This contain contains reporting functionality in development used by jupyter notebooks directly
or by more than one of the more specialized dev plotting modules, such as plots_dev_mhc and
plots_dev_ptm.  The stuff that ends up in here is probably used by *both* of those, or
is called directly by a notebook but is too "in development" to be included in plots.py
"""


def _run_labels(run_name):
    return "_".join(run_name.split("_")[:-1])


def _run_iz_count_pep_iz(df):
    """
    Given a df that has many pep_i that each belong to some run_i,
    return a list of run_i reverse-sorted for the number of pep_i.
    I.e. the first elem of this list is the run_i that has
    the most pep_i.

    Several parallel lists are returned, each as values in a Munch:

    run_iz: the run_i sorted by who has the most peptides
    run_labels: the run_name, with hash suffix removed
    pep_counts: the peptide count for each run
    peps: a list of pep_i for each run
    """

    grp = df.groupby("run_i")
    run_iz = np.array(list(grp.indices.keys()))
    peps = np.array([g.pep_i.values for r, g in grp])
    pep_counts = grp.pep_i.count().values
    run_labels = np.array([_run_labels(g.run_name.iloc[0]) for r, g in grp])
    pep_counts_reverse_sort_i = np.argsort(pep_counts)[::-1]
    return Munch(
        run_iz=run_iz[pep_counts_reverse_sort_i],
        run_labels=run_labels[pep_counts_reverse_sort_i],
        pep_counts=pep_counts[pep_counts_reverse_sort_i],
        peps=peps[pep_counts_reverse_sort_i],
    )


def peps_prec_at_min_recall_df(peps_prs_df, min_recall=0.1):
    """
    Given a df containing prs for one or more peptides, find the best precision
    for each peptide that meets the min_recall requirement.  If it cannot be
    met (for a given peptide), return None.

    Though the information here all deals in precision-recall, it
    seems strange to place this in CallBag whose business is computing PR curves.
    If this code were placed there, it would be @staticmethod.

    peps_prs_df: a df containing a set of prec,recall scores for each pep_i
    min_recall: the recall threshold to apply in filtering results

    returns: a df containing one row for each pep_i, which also includes
             the best precision available at the requested min_recall, or
             fallback values as stipulated above in the case this threshold
             cannot be met for some peptide(s).
    """

    def best_prec_at_recall(df):
        df = df.sort_values("recall", ascending=False).reset_index(drop=True)
        precs = df[df.recall >= min_recall].prec
        # Returning best precision that meets requested recall threshold.
        if len(precs) > 0:
            return df.loc[precs.idxmax()]

        # There is no entry in which recall > min_recall.
        # Return the best recall available, which may get discarded later.
        return df.loc[0]

    df = peps_prs_df.groupby("pep_i").apply(best_prec_at_recall)
    df = df[df.recall >= min_recall]
    return df.sort_values(by=["prec", "recall"], ascending=[False, False]).reset_index(
        drop=True
    )


def plot_confusion_matrix_compare(
    run,
    pep_i,
    score_threshold,
    classifier=None,
):
    """
    Plot two confusion matrices - one with all calls, and one with calls culled
    according to score_threshold.  Display precision and recall for pep_i
    in the title of the comparison plots.

    classifier: None to use any available preferred classifier, or one of the
                supported classifiers in RunResult::get_available_classifiers(),
                e.g. 'rf', 'nn'

    """
    cb = run.test_call_bag(classifier=classifier)
    z = ZPlots()
    with z(_cols=2, f_x_axis_label="true pep_i"):

        def pr(cm, p_i):
            prec = np_safe_divide(cm[p_i, p_i], np.sum(cm[p_i, :]))
            recall = np_safe_divide(cm[p_i, p_i], np.sum(cm[:, p_i]))
            return prec, recall

        conf_mat = cb.conf_mat()
        prec, recall = pr(conf_mat, pep_i)
        z.im(
            np.array(conf_mat),
            _size=500,
            f_title=f"{run.run_name}: pep_i={pep_i} precision={prec:.2f} recall={recall:.2f} {cb.classifier_name}",
            f_y_axis_label="pep_i   predicted by classifier",
        )

        conf_mat = cb.conf_mat_at_score_threshold(score_threshold)
        prec, recall = pr(conf_mat, pep_i)
        z.im(
            np.array(conf_mat),
            _size=500,
            f_title=f"precision={prec:.2f} recall={recall:.2f} score={score_threshold:.2f} {cb.classifier_name}",
        )


def plot_flu_info(run, flu, peps_prs_df, min_recall=0.005, classifier=None):
    """

    flu: string fluorosequence e.g. '..1......;0,0,0'

    peps_prs_df: the DataFrame from a notebook containing precision,recall,score
                 for the peptides you're interested in.

    classifier: None to use any available preferred classifier, or one of the
                supported classifiers in RunResult::get_available_classifiers(),
                e.g. 'rf', 'nn'
    """
    hd("h2", f"flu_info for: {flu}")

    cb = run.test_call_bag(classifier=classifier)
    pf2 = cb.peps__pepstrs__flustrs__p2()
    pf2 = pf2[pf2.flustr == flu]
    pep_iz = pf2.pep_i.values
    print(f"Peptides: {pep_iz}")

    if peps_prs_df.empty:
        # The peps_prs_df that is passed to us is typically the large
        # DataFrame with PRS information for ALL peptides across ALL
        # runs, but if this is a PTM notebook, it is often the case
        # that PRS info is only loaded for peptides with PTM locations.
        # This DataFrame has lots of other information, which is what
        # we actually want to display, otherwise we'd just recompute
        # the PRS for these peptides here.
        print("No PR info found for peps. (No PTM peps with this flu?)")
    else:
        df_pr = peps_prec_at_min_recall_df(peps_prs_df, min_recall=min_recall)
        display(df_pr)

    # See the fn in plots_dev_mhc which calls the lower-level plot_pr_curve using the
    # prs info we already have in the df.  The following call recomputes the PR all over
    # again for these peps!
    plots.plot_pr_breakout(
        run,
        pep_iz=pep_iz,
        _size=500,
        f_title=f"PR for peptides with flu {flu} ({cb.classifier_name})",
    )
