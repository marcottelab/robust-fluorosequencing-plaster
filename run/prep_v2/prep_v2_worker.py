import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from plaster.run.prep.c import c_prep
from plaster.run.prep_v2.prep_v2_params import PrepV2Params, Protein
from plaster.run.prep_v2.prep_v2_result import PrepV2Result
from plaster.tools.aaseq import proteolyze
from plaster.tools.aaseq.aaseq import aa_str_to_list
from plaster.tools.c_common.c_common_tools import DytType
from plaster.tools.zap import zap


def _proteolyze(pro_seq_df, proteases):
    """
    Break a single protein (one aa per row of pro_seq_df) into peptide fragments
    using the proteases, which may be None, a single protease name, or a list of
    protease names.

    This is called by an groupby("pro_i").apply(...)

    Returns:
        A DF where each row is:
            (aa, pro_pep_i, pep_offset_in_pro)

    TODO: Implement including missed cleavages
    """

    n_aas = len(pro_seq_df)

    if proteases is not None:
        if type(proteases) is not list:
            proteases = [proteases]
        cleave_before_iz = []
        for protease in proteases:
            assert protease is not None
            rules = proteolyze.compile_protease_rules(protease)
            cleave_before_iz += proteolyze.cleavage_indices_from_rules(
                pro_seq_df, rules
            )
        cleave_before_iz = sorted(list(set(cleave_before_iz)))
        start_iz = np.append([0], cleave_before_iz).astype(int)
        stop_iz = np.append(cleave_before_iz, [n_aas]).astype(int)
    else:
        start_iz = np.array([0]).astype(int)
        stop_iz = np.array([n_aas]).astype(int)

    # pro_i = pro_seq_df.pro_i.iloc[0]
    rows = [
        (pro_seq_df.aa.values[offset], pep_i, offset)
        for pep_i, (start_i, stop_i) in enumerate(zip(start_iz, stop_iz))
        for offset in range(start_i, stop_i)
    ]
    return pd.DataFrame(rows, columns=["aa", "pro_pep_i", "pep_offset_in_pro"])


def _step_1_check_for_uniqueness(pro_spec_df):
    # STEP 1: Check for uniqueness
    dupe_seqs = pro_spec_df.sequence.duplicated(keep="first")
    if dupe_seqs.any():
        print("The following sequences are duplicated")
        for d in pro_spec_df[dupe_seqs].itertuples():
            print(f"{d.name}={d.sequence}")
        raise ValueError("Duplicate protein seq(s)")

    dupe_names = pro_spec_df.name.duplicated(keep="first")
    if dupe_names.any():
        # TASK: Make a better error enumerating the duplicates
        raise ValueError("Duplicate protein name(s)")


def _step_2_create_pros_and_pro_seqs_dfs(pro_spec_df, use_zap=False):
    """
    Create pros_df and pro_seqs_df.
    Converts the sequence as a string into normalzied DataFrames
    """

    # Sort proteins such that the protein(s) being 'is_poi' are at the top, which means
    # the most interesting peptides start at pep_i==1.
    _pro_spec_df = pro_spec_df.sort_values(by=["is_poi", "name"], ascending=False)

    if use_zap:
        pro_lists = zap.arrays(
            aa_str_to_list, dict(seqstr=_pro_spec_df.sequence.values)
        )
    else:
        pro_lists = map(aa_str_to_list, _pro_spec_df.sequence.values)

    # Make a full-df with columns "aa", "pro_i", "pro_name", and "ptm_locs", "is_poi"
    # Then split this into the two fully normalized dfs
    df = pd.DataFrame(
        [
            (i, pro_i + 1, pro_name, pro_ptm_locs, is_poi)
            for pro_i, (pro, pro_name, pro_ptm_locs, is_poi) in enumerate(
                zip(
                    pro_lists,
                    _pro_spec_df.name,
                    _pro_spec_df.ptm_locs,
                    _pro_spec_df.is_poi,
                )
            )
            for i in pro
        ],
        columns=["aa", "pro_i", "pro_name", "pro_ptm_locs", "is_poi"],
    )

    # ADD reserved nul row
    nul = pd.DataFrame(
        [dict(aa=".", pro_i=0, pro_name="nul", pro_ptm_locs="", is_poi=False)]
    )
    df = pd.concat((nul, df))

    pros_df = (
        df[["pro_i", "pro_name", "pro_ptm_locs", "is_poi"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .rename(columns=dict(pro_name="pro_id"))
    )
    pros_df["pro_is_decoy"] = False

    pro_seqs_df = df[["pro_i", "aa"]].reset_index(drop=True)

    return pros_df, pro_seqs_df


def _step_3_generate_decoys(pros_df, pro_seqs_df, decoy_mode, shuffle_n=1):

    decoy_modes = [None, "none", "reverse", "shuffle"]
    if decoy_mode not in decoy_modes:
        raise NotImplementedError

    if decoy_mode is None or decoy_mode == "none":
        return (
            pd.DataFrame([], columns=PrepV2Result.pros_columns),
            pd.DataFrame([], columns=PrepV2Result.pro_seqs_columns),
        )

    smallest_pro_i = pros_df.pro_i.min()
    assert smallest_pro_i == 0

    if decoy_mode == "reverse":

        n_pros = pros_df.pro_i.max() + 1

        # Skip the nul
        pros_df = pros_df.set_index("pro_i").loc[1:]
        pro_seqs_df = pro_seqs_df.set_index("pro_i").loc[1:]

        def reverse_seq(x):
            x["aa"] = x["aa"].iloc[::-1].values
            return x

        decoy_seqs_df = (
            pro_seqs_df.groupby("pro_i", sort=False).apply(reverse_seq).reset_index()
        )
        decoy_seqs_df.pro_i += n_pros - 1  # -1 because we are not counting the nul

        pro_seqs_df = pro_seqs_df.reset_index()
        pros_df = pros_df.reset_index()

        def reverse_pro_ptm_locs():
            reversed_locs = []
            for pro in pros_df.itertuples():
                seq_length = pro_seqs_df[pro_seqs_df.pro_i == pro.pro_i].count()[0]
                reversed = ";".join(
                    map(
                        str,
                        sorted(
                            [
                                seq_length - int(x) + 1
                                for x in (
                                    pro.pro_ptm_locs
                                    if pro.pro_ptm_locs is not None
                                    else ""
                                ).split(";")
                                if x
                            ]
                        ),
                    )
                )
                reversed_locs += [reversed]
            return reversed_locs

        decoys_df = pd.DataFrame(
            dict(
                pro_i=np.arange(n_pros, 2 * n_pros - 1),  # -1 to skip the nul
                pro_is_decoy=[True] * (n_pros - 1),  # same
                pro_id=[f"rev-{i}" for i in pros_df.pro_id],
                pro_ptm_locs=reverse_pro_ptm_locs(),
            )
        )

    else:

        if list(pros_df.pro_ptm_locs.unique()) != [""]:
            raise NotImplementedError("shuffle decoys don't support ptm locations")

        rng = np.random.default_rng()

        def shuffle_seq(x, decoy_pro_i):
            xcopy = x.copy()
            rng.shuffle(xcopy.aa.values)
            xcopy.pro_i = decoy_pro_i
            return xcopy

        shuffle_groups = []
        decoy_pro_i = pro_seqs_df.pro_i.max()
        decoy_pro_i_range = []
        decoy_pro_ids = []

        # Skip the nul
        pros_df_lookup = pros_df.set_index("pro_i")

        for group in pro_seqs_df.groupby("pro_i", sort=False):
            this_pro_i = group[1].pro_i.unique()
            if this_pro_i != 0:
                for _ in range(shuffle_n):
                    decoy_pro_i = decoy_pro_i + 1
                    decoy_pro_i_range.append(decoy_pro_i)
                    decoy_pro_ids.append(
                        f"{decoy_pro_i}:shu-{pros_df_lookup.loc[group[0], 'pro_id']}"
                    )
                    shuffle_groups.append(shuffle_seq(group[1], decoy_pro_i))

        decoy_seqs_df = pd.concat(shuffle_groups).reset_index(drop=True)

        decoys_df = pd.DataFrame(
            dict(
                pro_i=decoy_pro_i_range,  # -1 to skip the nul
                pro_is_decoy=[True] * len(decoy_pro_i_range),  # same
                pro_id=decoy_pro_ids,
                pro_ptm_locs="",
            )
        )

    return decoys_df, decoy_seqs_df


def _step_4_proteolysis(pro_seqs_df, proteases):

    # TASK: Need ot fix this parallelization code...
    # pep_dfs = parallel_groupby_apply(pro_seqs_df.groupby("pro_i"), _proteolyze, compiled_protease_rules=compiled_protease_rules, _process_mode=True)
    # peps_df = pd.concat(pep_dfs)
    # peps_df = peps_df.reset_index(level=0).set_index(["pro_i", "pro_pep_i"])

    peps_df = (
        pro_seqs_df.groupby("pro_i")
        .apply(_proteolyze, proteases=proteases)
        .reset_index(level=0)
        .set_index(["pro_i", "pro_pep_i"])
    )

    # At this point the peps_df has a "local" index for the peptide -- ie. it restarts
    # every protein. But we want to concatenate all these into one big list of peptides.
    # The peps_df is indexed by "pro_i", "pro_pep_i" so a conversion table
    # is built "pro_pep_to_pep_i" and then this is merged with the peps_df causing
    # the "global" pep_i sequence to get into the pep_seqs_df and the "pro_pep_i"
    # can then be dropped.

    pro_pep_to_pep_i = peps_df.index.unique().to_frame().reset_index(drop=True)
    pro_pep_to_pep_i = pro_pep_to_pep_i.rename_axis("pep_i").reset_index()
    pep_seqs_df = pd.merge(
        left=pro_pep_to_pep_i, right=peps_df, on=["pro_pep_i", "pro_i"]
    ).drop(columns="pro_pep_i")

    # SET the pep_start and pep_stop based on the pep_offset_in_pro min and max of each pep_i group
    peps_df = pep_seqs_df.reset_index()[
        ["pro_i", "pep_i", "pep_offset_in_pro"]
    ].drop_duplicates()
    peps_df["pep_start"] = peps_df.groupby(["pep_i"]).pep_offset_in_pro.transform("min")
    peps_df["pep_stop"] = (
        peps_df.groupby(["pep_i"]).pep_offset_in_pro.transform("max") + 1
    )
    peps_df = (
        peps_df.drop(columns="pep_offset_in_pro")
        .drop_duplicates()
        .reset_index(drop=True)
    )[PrepV2Result.peps_columns]
    # [PrepV2Result.peps_columns] to reorder to canonical order avoiding warnings on concat etc.

    return peps_df, pep_seqs_df[["pep_i", "aa", "pep_offset_in_pro"]]


def _do_ptm_permutations(df, n_ptms_limit):
    """

    Apply the PTM permuations which are the ways that PTMS might
    or might not be present at a set of locations.

    Example:
        Suppose positions [2, 4, 10] are modifiable in a protein
        then the PTM might be present at all of the combinations:
        [ (2), (4), (10), (2,4), (2,10), (4,10), (2,4,10) ]

    Args:
        df: DF with a single (shared values) for:
                pep_i, and pro_ptm_locs, and pep_offset_in_pro
            as well as an aa for each location in the peptide.
            Example:
                pep_i aa  pep_offset_in_pro     pro_ptm_locs
                1     A   0                     1;4
                1     B   1                     1;4
                1     C   2                     1;4
                1     D   3                     1;4
                1     E   4                     1;4
                1     F   5                     1;4
    """

    assert list(
        df.dtypes.loc[["aa", "pep_i", "pep_offset_in_pro", "pro_ptm_locs"]]
    ) == ["object", "int64", "int64", "object"]

    # pro_ptm_locs should be identical for all rows
    pro_ptm_locs = df.pro_ptm_locs.values[0]
    if not pro_ptm_locs:
        return []

    # get 0-based indices from string representation which is actually 1-based;
    # these are for the entire protein.
    ptm_locs_zero_based = [(int(x) - 1) for x in pro_ptm_locs.split(";")]

    # LIMIT to the ptms that coincide with the range spanned by this peptide.
    min_pos = df.pep_offset_in_pro.min()
    max_pos = df.pep_offset_in_pro.max()
    ptm_locs_zero_based = [x for x in ptm_locs_zero_based if min_pos <= x <= max_pos]

    n_ptms = len(ptm_locs_zero_based)
    if n_ptms > n_ptms_limit:
        print(f"Skipping ptm for peptide {df.pep_i.iloc[0]} with {n_ptms} PTMs")

    if n_ptms_limit is not None and n_ptms > n_ptms_limit:
        return []

    ptm_combination_iz = [
        list(x)
        for length in range(1, len(ptm_locs_zero_based) + 1)
        for x in itertools.combinations(ptm_locs_zero_based, length)
    ]

    # ptm_combination_iz is now a list of lists. Example:
    #   ptm_locs_zero_based = [2,4,10]
    # generates:
    #   ptm_combination_iz = [ (2), (4), (10), (2,4), (2,10), (4,10), (2,4,10) ]
    #
    # The goal is to make a new peptide+seq for each of those index sets by
    # adding the modification '[p]' to the aa at that seq index location

    mod = "[p]"

    new_pep_seqs = []

    for ptm_locs in ptm_combination_iz:
        new_pep_seq = df.copy()

        new_pep_seq.pep_i = np.nan

        new_pep_seq = new_pep_seq.set_index("pep_offset_in_pro")
        new_pep_seq.loc[ptm_locs, "aa"] = new_pep_seq.loc[ptm_locs, "aa"] + mod
        new_pep_seq = new_pep_seq.reset_index()

        new_pep_seqs += [new_pep_seq]

    return new_pep_seqs


def _step_5_create_ptm_peptides(peps_df, pep_seqs_df, pros_df, n_ptms_limit):
    """
    Create new peps and pep_seqs by applying PTMs based on the pro_ptm_locs information
    in pros_df.
    """

    # 1. Get subset of proteins+peps with ptms by filtering proteins with ptms and joining
    # to peps and pep_seqs
    #

    # This None vs "" is messy.

    pros_with_ptms = pros_df[pros_df.pro_ptm_locs != ""]
    df = (
        pros_with_ptms.set_index("pro_i").join(peps_df.set_index("pro_i")).reset_index()
    )
    df = df.set_index("pep_i").join(pep_seqs_df.set_index("pep_i")).reset_index()

    if len(df) == 0:
        return None, None

    # 2. for each peptide apply _do_ptm_permutations which will result in
    # a list of new dataframes of the form joined above; new_pep_infos is a
    # list of these lists.
    with zap.Context(trap_exceptions=False, mode="thread"):
        new_pep_infos = zap.df_groups(
            _do_ptm_permutations,
            df.groupby("pep_i"),
            n_ptms_limit=n_ptms_limit,
        )

    # 3. create new peps, pep_seqs, from list of dfs returned in (2)
    #
    #    peps_columns = ["pep_i", "pep_start", "pep_stop", "pro_i"]
    #    pep_seqs_columns = ["pep_i", "aa", "pep_offset_in_pro"]
    #
    new_peps = []
    new_pep_seqs = []
    next_pep_i = peps_df.pep_i.max() + 1
    for new_peps_info in new_pep_infos:
        for pep_info in new_peps_info:
            # Note we only want one pep entry and pep_info contains enough rows to hold
            # the whole sequence for the peptide in the aa column.  So drop_duplicates()
            pep = pep_info[PrepV2Result.peps_columns].drop_duplicates()
            pep_seq = pep_info[
                PrepV2Result.pep_seqs_columns
            ].copy()  # avoid SettingWithCopyWarning with copy()

            pep.pep_i = next_pep_i
            pep_seq.pep_i = next_pep_i
            next_pep_i += 1

            new_peps += [pep]
            new_pep_seqs += [pep_seq]

    new_peps_df = pd.concat(new_peps)
    new_pep_seqs_df = pd.concat(new_pep_seqs)

    return new_peps_df, new_pep_seqs_df


def triangle_dytmat(n_cycles, n_dyes, include_multi_drop=False, include_nul_row=False):
    """
    Generate a "triangle" dytmat.
    This is used for photobleaching runs.

    Returns a dytmat

    Example: n_cycles = 3, n_dyes = 2
        0 0 0  # Nul row
        1 0 0
        1 1 0
        1 1 1
        2 0 0  # Multidrop
        2 1 0
        2 1 1
        2 2 0  # Multidrop
        2 2 1
        2 2 2
    """
    assert 1 <= n_dyes <= 4
    dytmat = []

    def _inner(row, cnt, cy_i):
        # function can't generate cnt < 1
        nonlocal dytmat
        for cy_j in range(cy_i + 1, n_cycles + 1):
            _row = row.copy()
            _row[cy_i:cy_j] = cnt
            dytmat += [_row]
            for cnt_i in range(cnt - 1, 0, -1):
                _inner(_row, cnt_i, cy_j)

    row = np.zeros((n_cycles,), dtype=DytType)
    if include_nul_row:
        dytmat += [row]
    for cnt_i in range(n_dyes, 0, -1):
        _inner(row, cnt_i, 0)

    dytmat = np.array(dytmat, dtype=DytType)

    if not include_multi_drop:
        multidrop = np.any(np.diff(dytmat, prepend=0, axis=1) < -1, axis=1)
        dytmat = dytmat[~multidrop]

    rev_cols = [dytmat[:, cy] for cy in range(dytmat.shape[1] - 1, -1, -1)]
    return dytmat[np.lexsort(rev_cols)]


def dyt_to_seq(dyt):
    """
    From 2211000 -> .X.X...
    Where "." is "." and X is standin amino-acid
    """
    assert len(dyt.shape) == 1
    diff = np.diff(dyt, append=0)
    assert np.all(diff <= 0)
    return np.where(diff == 0, ".", PrepV2Params._PHOTOBLEACHING_PSEUDO_AA)


def generate_photobleaching(n_cycles, n_dye_count):
    """
    CREATE pseudo peps and pros, 1 per dyt
    """
    # include_multi_drop needs to be False at moment because the
    # sim_v2 simulator isn't handling those correctly.
    # The correct solution would probably be to take the simulator
    # of the the picture completely and just pass in the dyetracks.
    dytmat = triangle_dytmat(
        n_cycles,
        n_dye_count,
        include_multi_drop=False,
        include_nul_row=True,
    )
    assert len(dytmat.shape) == 2

    n_dyts = dytmat.shape[0]

    # pros_df (pro_id, pro_is_decoy, pro_i, pro_ptm_locs, is_poi)
    pros_df = pd.DataFrame(
        dict(
            pro_id=[f"dyt_{i:03d}" for i in np.arange(n_dyts)],
            pro_is_decoy=0,
            pro_i=np.arange(n_dyts),
            pro_ptm_locs="",
            is_poi=False,
        )
    )

    # peps_df (pep_i, pep_start, pep_stop, pro_i)
    peps_df = pd.DataFrame(
        dict(
            pep_i=np.arange(n_dyts),
            pep_start=0,
            pep_stop=0,
            pro_i=np.arange(n_dyts),
        )
    )

    pep_seqs_df = c_prep.prep_dytmat_to_pep_seqs(dytmat)

    pro_seqs_df = pep_seqs_df[["pep_i", "aa"]].rename(columns=dict(pep_i="pro_i"))

    return pros_df, pro_seqs_df, peps_df, pep_seqs_df, dytmat


def prep_v2(prep_params: PrepV2Params, folder: Path):
    """
    Given protease and decoy mode, create proteins and peptides.

    Arguments:
        prep_params: PrepV2Params
        pro_spec_df: Columns: sequence (str), id (str), ptm_locs (str)
            can be None if prep_params.is_photobleaching_run

    Steps:
        1. Real proteins are checked for uniqueness in seq and id
        2. The real proteins are first string-split "unwound" into seq_ dataframes
           (one row per amino acid).
        3. The decoys are added by reversing those real DFs.
        4. The proteolysis occurs by a map against proteins
        5. PTMs are added

    ParamResults:
        Four DFs:
            * the pro data (one row per protein)
            * the pro_seq data (one row per aa) * n_pros
            * the pep data (one row per peptide)
            * the pep_seq data (one row per aa) * n_pres
    """

    # TODO: Clean up photobleaching logic
    photobleaching_dytmat = None
    if prep_params.is_photobleaching_run:
        # If is_photobleaching_run then we want to create
        # pseudo-peptides and proteins one for each possible dyetrack.
        (
            pros_df,
            pro_seqs_df,
            peps_df,
            pep_seqs_df,
            photobleaching_dytmat,
        ) = generate_photobleaching(
            prep_params.photobleaching_n_cycles,
            prep_params.photobleaching_run_n_dye_count,
        )

    else:

        # if prep_params.drop_duplicates:
        #     pro_spec_df = pro_spec_df.drop_duplicates("sequence")
        #     pro_spec_df = pro_spec_df.drop_duplicates("name")

        _step_1_check_for_uniqueness(prep_params.protein_df)

        reals_df, real_seqs_df = _step_2_create_pros_and_pro_seqs_dfs(
            prep_params.protein_df
        )

        decoys_df, decoy_seqs_df = _step_3_generate_decoys(
            reals_df,
            real_seqs_df,
            prep_params.decoy_mode,
            shuffle_n=prep_params.shuffle_n,
        )

        pros_df = pd.concat((reals_df, decoys_df), sort=True).reset_index(drop=True)
        pros_df = pros_df.astype(dict(pro_i=int))

        pro_seqs_df = pd.concat((real_seqs_df, decoy_seqs_df)).reset_index(drop=True)

        peps_df, pep_seqs_df = _step_4_proteolysis(pro_seqs_df, prep_params.proteases)

        # if prep_params.n_peps_limit > -1:
        #     # This is used for debugging to limit the number of peptides.
        #     # This draws randomly to hopefully pick up decoys too
        #     n_peps = peps_df.pep_i.nunique()
        #     pep_iz = np.sort(
        #         np.random.choice(n_peps, prep_params.n_peps_limit, replace=False)
        #     )
        #     pep_iz[0] = 0  # Ensure the reserved value is present
        #     peps_df = peps_df.loc[pep_iz]
        #     pep_seqs_df = pep_seqs_df[pep_seqs_df.pep_i.isin(pep_iz)]

        # if prep_params.n_ptms_limit > -1:
        #     # n_ptms_limit can be a non-zero value to limit the number of ptms
        #     # allowed per peptide, or set to 0 to skip ptm permutations even when
        #     # there are PTMs annotated for the proteins in protein_csv.
        #     ptm_peps_df, ptm_pep_seqs_df = _step_5_create_ptm_peptides(
        #         peps_df, pep_seqs_df, pros_df, prep_params.n_ptms_limit
        #     )
        #     if ptm_peps_df is not None and len(ptm_peps_df) > 0:
        #         peps_df = pd.concat([peps_df, ptm_peps_df])
        #         pep_seqs_df = pd.concat([pep_seqs_df, ptm_pep_seqs_df])

    result = PrepV2Result(
        params=prep_params,
        _pros=pros_df,
        _pro_seqs=pro_seqs_df,
        _peps=peps_df,
        _pep_seqs=pep_seqs_df,
        photobleaching_dytmat=photobleaching_dytmat,
    )

    result.set_folder(folder)

    return result


def photobleaching_fixture(n_cycles, n_dye_count, folder):
    prep_params = PrepV2Params(
        is_photobleaching_run=True,
        photobleaching_n_cycles=n_cycles,
        photobleaching_run_n_dye_count=n_dye_count,
        proteins=[
            Protein(
                name="pb_fixture",
                sequence="",
                ptm_locs="",
                is_poi=0,
                abundance=0,
            )
        ],
    )
    return prep_v2(prep_params=prep_params, folder=folder)
