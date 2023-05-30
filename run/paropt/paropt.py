"""
Linear unmixing parameter optimization code

https://docs.google.com/document/d/1Zpt08J-dt5k8JlVFGuvbbteP_WlBwer5Xq47lO0sh4g/edit#heading=h.r749d7a95qj1

See plaster/explore/paropt_optimize_demo.ipynb for examples of how to use this code
See plaster/explore/paropt_classify_demo.ipynb for examples of how to prepare data for this code
"""
import multiprocessing as mp
from collections import Counter
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import scipy
import structlog

import plaster.genv2.generators.sim as simgen
import plaster.reports.helpers.report_params_filtering as pf
import plaster.run.sim_v3.sim_v3_worker as sim_v3_worker
from plaster.reports.helpers.report_params import FilterParams
from plaster.run.ims_import.ims_import_result_v1 import ImsImportResult
from plaster.run.prep_v2.prep_v2_worker import prep_v2
from plaster.run.rf_v2.rf_v2_result import RFV2ResultDC
from plaster.run.sigproc_v2.sigproc_v2_result import SigprocV2ResultDC
from plaster.run.sim_v3.sim_v3_result import SimV3Result

logger = structlog.getLogger()

# TODO: convert this to dataclass using sim_v3_params dataclasses
# setup_info example for use with state_setup
# dict with keys "markers", "seq_params", and "paropt_params"
# the value of each is a dict as follows

# markers : a dict of dicts where the keys are single characters
# and the values are dicts containing the following fields
# - n        : maximum number of this label present on the spanning peptides
# - p_bleach : per cycle bleaching + chemical destruction rate for this label
# - p_dud    : initial dud rate for this label

# seq_params : dict containing the following key and value pairs:
# - n_edmans        : number of edman degradation cycles
# - n_residues      : number of residues in each peptide
# - p_cyclic_block  : per cycle probability of permanent damage to the peptide
#                     preventing further Edman degradation
# - p_detach        : per cycle probability of peptide loss from slide
# - p_edman_failure : per cycle probability of Edman failure

# paropt_params : dict containing the following key and value pairs:
# - n_samples : number of samples to use for dytmat MC simulation
# - method    : whether to run sims in "forward" or "backward" mode

SETUP_EXAMPLE = {
    "markers": {
        "A": {"n": 1, "p_bleach": 0.05, "p_dud": 0.2},
        "B": {"n": 1, "p_bleach": 0.05, "p_dud": 0.2},
    },
    "seq_params": {
        "n_edmans": 8,
        "n_residues": 10,
        "p_cyclic_block": 0.1,
        "p_detach": 0.01,
        "p_edman_failure": 0.3,
    },
    "paropt_params": {
        "method": "backward",
        "n_samples": 5000,
    },
}


def dytmat2txt(dytmat: np.array) -> list[str]:
    """Convert a SimV3Result dytmat array to a list of dyetrack strings."""
    return ["".join(map(str, dytmat[k, :])) for k in range(1, dytmat.shape[0])]


def translate_dyt_index(
    span_target: list[str], query_dytmat: np.ndarray
) -> tuple[list[str], list[str]]:
    """
    Given a spanning list of dyetrack strings in span_target and a
    SimV3Result dytmat array in query_dytmat, return the intersecting indices of
    the spanning list of dyetrack strings and the dytmat array.

    Used to index dytmat_sim results into a spanning set of dyetracks when producing a
    dyetrack transition matrix.

    Args
    ----
    span_target  : Spanning list of dyetrack strings generated by a classifier run with
                   classify_dyetracks and span_dyetracks
    query_dytmat : SimV3Result dytmat

    Returns
    -------
    Tuple containing two equal-sized lists of matching dyetrack string indexes from the
    span_target and query_dytmat, respectively.
    """
    query = dytmat2txt(query_dytmat)
    _, span_ind, query_ind = np.intersect1d(span_target, query, return_indices=True)
    return span_ind, query_ind


def generate_transition_matrix(dytpeps: np.ndarray) -> np.ndarray:
    """
    Given a sim_v3_result dytpeps array, return a dyetrack transition matrix that
    transforms a peptide distribution vector into a dyetrack distribution vector
    based on the dyetrack-peptide mapping specified in the dytpeps array.

    Args
    ----
    dytpeps : SimV3Result dytpeps

    Returns
    -------
    A dyetrack transition matrix of size n_dyetracks x n_peptides where
    n_dyetracks is the number of dyetracks generated by the peptides in dytpeps
    and n_peptides is the number of peptides in dytpeps minus the empty peptide.

    """
    dyt_max, pep_max = dytpeps[:, :2].max(axis=0)
    A_mat = np.zeros((dyt_max, pep_max))

    for pep_idx in range(1, pep_max + 1):
        if pep_idx > 0:
            tmp = dytpeps[dytpeps[:, 1] == pep_idx, :]
            argsort_idx = np.argsort(tmp[:, 0])
            tmp = tmp[argsort_idx, :].astype(float)
            tmp[:, 2] = tmp[:, 2] / tmp[:, 2].sum()
            for k in range(tmp.shape[0]):
                A_mat[int(tmp[k, 0]) - 1, int(tmp[k, 1]) - 1] = tmp[k, 2]
    return A_mat


def span_peptides(queue: list[str], n_residues: int) -> list[str]:
    """
    Given a list of single character amino acid labels queue and a total
    number of peptide residues n_residues, create all possible peptides of
    length n_residues containing some or all of the amino acid labels in queue.

    Examples:
    queue      : ['A', 'B']
    n_residues : 4
    result     : ['....', 'A...', 'B...', 'AB..', 'BA..', 'A.B.', 'B.A.', ...]

    queue      : ['A', 'A']
    n_residues : 4
    result     : ['....', 'A...', 'AA..', 'A.A.', 'A..A', '.A..', '.AA.', ...]

    queue      : ['A', 'A', 'B']
    n_residues : 4
    result     : ['....', 'A...', 'B...', 'AA..', 'AB..', 'AAB.', 'ABA.', ...]

    Args
    ----
    queue      : List of single character amino acids describing how many
                 labels are to be used in the peptide span. Repeating a letter
                 will add a second, third, etc label of that letter's type.
    n_residues : Number of residues in the peptides

    Returns
    -------
    List of strings containing all possible peptides of length n_residues
    containing some or all of the amino acid labels in queue.

    """
    blank = ["." for x in range(n_residues)]
    work = []
    done = []
    added_letter = set()
    for letter in queue:
        if letter not in added_letter:
            work.append(blank)
        added_letter.add(letter)
        work.extend(map(list, set(map(lambda x: "".join(x), done))))
        while work:
            todo = work.pop()
            # done.append(todo)
            for loc in range(len(todo)):
                if todo[loc] == ".":
                    add = todo.copy()
                    add[loc] = letter
                    done.append(add)
    return sorted(set(map(lambda x: "".join(x), done)))


def state_setup(setup_info: dict[str, Any]) -> tuple[list[float], list[tuple[float]]]:
    """
    Given setup_info (see SETUP_EXAMPLE above), set up a state vector
    containing parameters laid out for the paropt problem.

    Args
    ----
    setup_info : Dict containing paropt info as described in SETUP_EXAMPLE

    Returns
    -------
    parameters : List of float parameters for the paropt problem
    bounds     : List of float-pairs as bounds on the search
    """
    # set up X_start in the correct order
    x = []
    for aa in setup_info["markers"]:
        x.append(setup_info["markers"][aa]["p_bleach"])
        if len(setup_info["markers"]) > 1:
            x.append(setup_info["markers"][aa]["p_dud"])
        else:
            # For one-count data, detach and dud are indistinguishable.
            # To reduce parameter search space, we set the range and parameter
            # to zero.
            x.append(0.0)
    x.append(setup_info["seq_params"]["p_cyclic_block"])
    x.append(setup_info["seq_params"]["p_detach"])
    x.append(setup_info["seq_params"]["p_edman_failure"])

    bounds = []
    r = (0.0, 1.0)
    for aa in setup_info["markers"]:
        bounds += [r]
        if len(setup_info["markers"]) > 1:
            bounds += [r]
        else:
            # See previous comment about dud-detach for one-count data.
            bounds += [(0.0, 0.0)]
    bounds += [r] * 3

    assert len(x) == len(bounds)

    return x, bounds


def protein_list_setup(setup_info: dict[str, Any]) -> list[simgen.Protein]:
    """
    Given setup_info, use span_peptides to set up the spanning protein list needed to
    construct a SimConfig consisting of all peptides of n_residues length containing
    some or all of the labels specified in setup_info.

    If error parameters are set to 0, a peptide list generated
    with protein_list_setup can be used to generate dyetracks suitable for
    classify_dyetracks instead of using span_dyetracks. The peptides generated
    will span the space of possible dyetracks for the chosen queue and n_residues.

    Args
    ----
    setup_info : Dict containing paropt info as described in SETUP_EXAMPLE

    Returns
    -------
    List of simgen.Protein entities

    """
    marker_queue = []
    for aa in setup_info["markers"]:
        for _ in range(setup_info["markers"][aa]["n"]):
            marker_queue.append(aa)
    return [
        simgen.Protein(
            name=f"PEP{idx}", sequence=pep, abundance=1.0, is_poi=True, ptm_locs=None
        )
        for idx, pep in enumerate(
            span_peptides(marker_queue, setup_info["seq_params"]["n_residues"])
        )
    ]


def gen_config(
    X: list[float],
    seed: int,
    setup_info: dict[str, Any],
    protein_list: list[simgen.Protein],
) -> simgen.SimConfig:
    """
    Given a parameter state vector, a random seed, setup_info, and a protein_list,
    construct a SimConfig usable for dytmat simulation.

    Args
    ----
    X            : parameter state vector
    seed         : random seed
    setup_info   : Dict containing paropt info as described in SETUP_EXAMPLE
    protein_list : List of simgen.Protein entities

    Returns
    -------
    SimConfig constructed from the input arguments

    """
    markers = [
        simgen.Marker(
            aa=umark,
            gain_mu=1.0,
            gain_sigma=0.0,
            bg_sigma=0.0,
            p_bleach=X[idx * len(setup_info["markers"])],
            p_dud=X[(idx * len(setup_info["markers"])) + 1],
        )
        for idx, umark in enumerate(setup_info["markers"])
    ]
    seq_params = simgen.SeqParams(
        p_initial_block=0.0,
        p_label_failure=0.0,
        p_cyclic_block=X[-3],
        p_detach=X[-2],
        p_edman_failure=X[-1],
        n_edmans=setup_info["seq_params"]["n_edmans"],
        row_k_sigma=0.0,
    )

    config = simgen.SimConfig(
        type="sim",
        job="/dev/null",
        markers=markers,
        seq_params=seq_params,
        proteins=protein_list,
        n_samples_train=setup_info["paropt_params"]["n_samples"],
        n_samples_test=0,
        decoy_mode="none",
        seed=seed,
    )

    return config


def gen_pep_info(
    start_config: simgen.SimConfig, true_pep_start: str
) -> tuple[pd.DataFrame, int]:
    """
    Given a starting SimConfig start_config and a string uniquely identifying the start of the
    peptide under study, return the pepseqs dataframe used to generate PCBs and the peptide vector
    index of the peptide under study.

    Args
    ----
    start_config   : SimConfig constructed from the input arguments
    true_pep_start : Start of the true peptide under study. For JSP211 with
                     SETUP_EXAMPLE this is .A..B, ie label A in 2nd AA position
                     and label 1 in 5th AA position

    Returns
    -------
    Two element tuple, first element the pepseqs dataframe derived from start_config
    and second element the peptide vector index of the specified peptide under study.

    Notes
    -----
    The peptide vector index does not include the null peptide, so we subtract 1
    from the index found by our search.

    """
    prep_params = simgen.generate_prep_params(config=start_config)
    prep_result = prep_v2(prep_params=prep_params, folder="/dev/null")
    pros_peps_pepstrs = prep_result.pros__peps__pepstrs()

    pepseqs = prep_result.pepseqs__with_decoys()
    # subtract 1 here because we don't include the null peptide
    true_pep_idx = [
        idx
        for idx in pros_peps_pepstrs.index
        if pros_peps_pepstrs.loc[idx, "seqstr"].startswith(true_pep_start)
    ][0] - 1

    return pepseqs, true_pep_idx


def paropt_objfun(
    setup_info: dict[str, Any],
    protein_list: list[simgen.Protein],
    pepseqs: pd.DataFrame,
    true_pep_idx: int,
    score_counts: pd.DataFrame,
    X: list[float],
    seed: int,
    accum: list[Any],
) -> float:
    """
    Evaluate a linear unmixing result using the forward or backward method. Intended
    to be partialized to a function that keeps only the final three arguments
    X, seed, and accum.

    Args
    ----
    setup_info   : Dict containing paropt info as described in SETUP_EXAMPLE
    protein_list : List of simgen.Protein entities
    pepseqs      : pepseqs dataframe from gen_pep_info
    true_pep_idx : true_pep_idx from gen_pep_info
    score_counts : dataframe containing dyetrack classification information
    X            : parameter state vector
    seed         : random seed
    accum        : List that aggregates evaluation outcomes

    Returns
    -------
    Objective function value

    Side effects
    ------------
    Appends tuples to accum consisting of the parameter state value, the
    objective function value, and the two components of the objective value
    that are the square root of the non-target peptide component of the
    predicted peptide vector and the error of the non-negative linear
    least squares calculation.

    """
    method = "backward"
    if (
        "paropt_params" in setup_info
        and "method" in setup_info["paropt_params"]
        and setup_info["paropt_params"]["method"] == "forward"
    ):
        method = "forward"

    current_config = gen_config(X, seed, setup_info, protein_list)
    sim_params = simgen.generate_sim_params(config=current_config)
    synth_pcbs = sim_params.pcbs(pepseqs)

    dytmat, dytpeps, _ = sim_v3_worker._dytmat_sim(
        sim_params, synth_pcbs, sim_params.n_samples_train, use_zap=False
    )

    # Ignore the null dye track in count
    if len(dytmat) <= 1:
        logger.error("Parameters without dye tracks: ", X=X)
        if method == "forward":
            summary = (X, np.inf)
        else:
            summary = (X, np.inf, np.inf, np.inf)

        accum.append(summary)
        return np.inf

    dyt_labels = pd.Series(
        dytmat2txt(dytmat), name="dyt_labels", index=range(1, dytmat.shape[0])
    )

    source_ind, _ = translate_dyt_index(dyt_labels, dytmat)

    A_mat = generate_transition_matrix(dytpeps)
    B_mat = score_counts.loc[score_counts.index[source_ind], "counts"]

    if method == "forward":
        pure_X = np.zeros(len(protein_list))
        pure_X[true_pep_idx] = 1.0

        if A_mat.shape[1] != pure_X.shape[0]:
            logger.error(
                "Incompatible dimensions (forward)", A_mat=A_mat, pure_X=pure_X
            )
            obj = np.inf
        else:
            Y_pred = A_mat @ pure_X
            Y_real = B_mat
            Y_real /= Y_real.sum()

            obj = (((Y_pred - Y_real) / Y_real) ** 2.0).sum()

        summary = (X, obj)

    else:
        if A_mat.shape[0] != B_mat.shape[0]:
            logger.error("Incompatible dimensions (backward)", A_mat=A_mat, B_mat=B_mat)

            # Example message from nnls exhibiting incompatible dimensions failure:
            #   Incompatible dimensions. The first dimension of A is 3, while the shape of b is (7,)
            # https://github.com/scipy/scipy/blob/5e3d4676631b4d3abb85f3f25634072ce27f6e2a/scipy/optimize/_nnls.py#L68-L73

            summary = (X, np.inf, np.inf, np.inf)
            accum.append(summary)

            return np.inf

        opt_result = scipy.optimize.nnls(A_mat, B_mat)

        target = opt_result[0][true_pep_idx]
        norm_target = target / opt_result[0].sum()
        negsum_sqrt = (1.0 - norm_target) ** 0.5

        obj = negsum_sqrt * opt_result[1]

        summary = (X, obj, negsum_sqrt, opt_result[1])

    accum.append(summary)
    return obj


def peptide_prediction(
    setup_info: dict[str, Any],
    protein_list: list[simgen.Protein],
    pepseqs: pd.DataFrame,
    score_counts: pd.DataFrame,
    X: list[float],
    seed: int,
) -> tuple[np.array, float]:
    """
    Predict a peptide distribution from a dyetrack classification using the backward
    linear unmixing method.

    Args
    ----
    setup_info   : Dict containing paropt info as described in SETUP_EXAMPLE
    protein_list : List of simgen.Protein entities
    pepseqs      : pepseqs dataframe from gen_pep_info
    score_counts : dataframe containing dyetrack classification information
    X            : parameter state vector
    seed         : random seed

    Returns
    -------
    Two element tuple, where the first element is the predicted peptide distribution
    and the second element is the sum of squared residuals
    sum((A_mat @ X_pred - Y_exp)**2) where
    - A_mat is the dyetrack transition matrix
    - X_pred is the predicted peptide distribution
    - Y_exp is the classified dyetrack distribution

    """
    current_config = gen_config(X, seed, setup_info, protein_list)
    sim_params = simgen.generate_sim_params(config=current_config)
    synth_pcbs = sim_params.pcbs(pepseqs)

    dytmat, dytpeps, _ = sim_v3_worker._dytmat_sim(
        sim_params, synth_pcbs, sim_params.n_samples_train, use_zap=False
    )

    dyt_labels = pd.Series(
        dytmat2txt(dytmat), name="dyt_labels", index=range(1, dytmat.shape[0])
    )

    source_ind, _ = translate_dyt_index(dyt_labels, dytmat)

    A_mat = generate_transition_matrix(dytpeps)
    B_mat = score_counts.loc[score_counts.index[source_ind], "counts"]

    if A_mat.shape[0] != B_mat.shape[0]:
        logger.error("Incompatible dimensions", A_mat=A_mat, B_mat=B_mat)

        # Search for "Incompatible dimensions" in this file for more details.

        return np.inf

    opt_result = scipy.optimize.nnls(A_mat, B_mat)

    return opt_result


def sp_paropt(
    input_args: tuple[Callable[..., float], list[float], int],
    maxiter: int = 5,
    maxfev: int = 5000,
    bounds: Optional[list] = None,
) -> list[Any]:
    """
    Attempt to minimize the specified paropt function with a given
    starting parameter vector and seed using Powell's method.
    Accumulate results into accum.

    Individual minimizations for a given seed may not be especially
    successful, so this is best thought of as a way to search through
    the parameter space by iterating over seeds and starting vectors.

    Args
    ----
    input_args : Three element tuple consisting of
                 - an optimization function to be minimized
                 - a parameter vector
                 - a random seed
    maxiter    : max Powell's iterations
    maxfev     : max function evaluations
    bounds     : List of float pairs for bounds on each parameter.

    Returns
    -------
    List contents of accum from the paropt function.

    """

    # Set disp:False to avoid messages like the following:
    # Optimization terminated successfully.
    #      Current function value: 28903.378548
    #      Iterations: 2
    #      Function evaluations: 170

    accum = []
    res = None
    try:
        paropt, X0, seed = input_args
        if bounds is None:
            bounds = [(0.0, 1.0) for _ in X0]

        res = scipy.optimize.minimize(
            paropt,
            X0,
            args=(seed, accum),
            method="Powell",
            bounds=bounds,
            options={
                "disp": False,
                "return_all": False,
                "maxiter": maxiter,
                "maxfev": maxfev,
            },
        )
    except Exception as e:
        logger.exception(e)

    return accum, res


def mp_paropt(
    setup_info: dict[str, Any],
    score_counts: pd.DataFrame,
    true_pep_start: str,
    seed_offset: int = 0,
    maxiter: int = 5,
    maxfev: int = 5000,
    nattempts: int = 0,
    bounds: Optional[list] = None,
    cb: Optional[Callable] = None,
    n_processes: int = -1,
) -> list[list[Any]]:
    """
    Given setup_info and score_counts, use multiple processors to search
    for best fit parameters to the linear unmixing problem.

    Individual minimizations for a given seed may not be especially
    successful, so this is best thought of as a way to search through
    the parameter space by iterating over seeds and starting vectors.

    Args
    ----
    setup_info     : Dict containing paropt info as described in SETUP_EXAMPLE
    score_counts   : dataframe containing dyetrack classification information
    seed_offset    : Shift all random seeds by this amount. Useful for exploration
    maxiter        : max Powell's iterations
    maxfev         : max function evaluations
    true_pep_start : Start of the true peptide under study. For JSP211 with
                        SETUP_EXAMPLE this is .A..B, ie label A in 2nd AA position
                        and label 1 in 5th AA position
    nattempts      : Number of independent runs. If <=0, use CPU count.
    bounds         : List of float pairs for bounds on each parameter.
    cb             : Progress callback function: cb(i, n, last_best_result)
    n_processes    : Number of concurrent processes, -1=use core count

    Returns
    -------
    List of accums, one per attempt. Each will contain a Powell's method
    minimization starting from a different random seed.
    """

    protein_list = protein_list_setup(setup_info)
    # Todo: Generate new initial guess X0 per attempt, within range.
    X0, bounds_ = state_setup(setup_info)
    if bounds is None:
        bounds = bounds_
    start_config = gen_config(X0, 0, setup_info, protein_list)
    pepseqs, true_pep_idx = gen_pep_info(start_config, true_pep_start)

    partial_paropt_objfun = partial(
        paropt_objfun, setup_info, protein_list, pepseqs, true_pep_idx, score_counts
    )

    if nattempts <= 0:
        nattempts = mp.cpu_count()

    paropt_args = [
        (partial_paropt_objfun, X0, x + seed_offset) for x in range(nattempts)
    ]

    accums = []
    ress = []
    thread_fn = partial(sp_paropt, maxiter=maxiter, maxfev=maxfev, bounds=bounds)

    logger.info("Optimization starting parameters", X0=X0, bounds=bounds)

    def cb_(accums):
        if cb is not None:
            try:
                last_best = sorted(accums[-1], key=lambda x: x[1])[0]
            except (IndexError, ValueError):
                last_best = None
            cb(len(accums), len(paropt_args), last_best)

    if n_processes == -1:
        n_processes = mp.cpu_count()

    if n_processes > 1:
        with mp.Pool(processes=n_processes) as pool:
            it = pool.imap_unordered(thread_fn, paropt_args, chunksize=1)
            while True:
                try:
                    accum, res = it.next(300)
                except mp.TimeoutError:
                    logger.warning("Timeout", len(accum), len(paropt_args))
                    break
                except StopIteration:
                    break
                accums.append(accum)
                ress.append(res)
                cb_(accums)
    else:
        for args in paropt_args:
            accum, res = thread_fn(args)
            accums.append(accum)
            ress.append(res)
            cb_(accums)

    return accums, ress


def score_counts_setup(
    ims_import_result: ImsImportResult,
    sigproc_result: SigprocV2ResultDC,
    sim_result: SimV3Result,
    rf_classify_result: RFV2ResultDC,
    filter_params: FilterParams,
) -> pd.DataFrame:
    """
    Given an ims_import_result and sigproc_result from a sigproc_v2 experimental run, a sim_result from a
    VFS classifier training run with classify_dyetracks=True, and an rf_classify_result from the
    classification of that experiment with classify_workflow, return a filtered score counts dataframe
    labeled with the dyetracks used in the classifier.

    Args
    ----
    ims_import_result  : from sigproc_v2 experimental processing
    sigproc_result     : from sigproc_v2 experimental processing
    sim_result         : from VFS classifier training with classify_dyetracks = True
    rf_classify_result : from classifier application to sigproc_v2 using classify_workflow
    """

    # set up score_counts_filter
    filter_df = pf.get_classify_default_filtering_df(
        sigproc_result, ims_import_result, filter_params, all_cycles=True
    )

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
    rfc_filter = rfc_df.loc[filter_df.pass_all, :]
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

    return score_counts_filter
