import random

import pandas as pd

from plaster.run.prep_v2 import prep_v2_worker
from plaster.run.prep_v2.prep_v2_params import PrepV2Params, Protein
from plaster.run.prep_v2.prep_v2_result import PrepV2Result
from plaster.tools.aaseq.aaseq import aa_random


def result_random_fixture(n_proteins):
    """
    Generate a fixture with randomly generate n_proteins
    """

    prep_params = PrepV2Params(
        proteins=[
            Protein(name=f"pro{i + 1}", sequence=aa_random(random.randrange(5, 100)))
            for i in range(n_proteins)
        ]
    )

    pro_spec_df = pd.DataFrame(
        dict(
            name=[f"pro{i + 1}" for i in range(n_proteins)],
            sequence=[aa_random(random.randrange(5, 100)) for i in range(n_proteins)],
            ptm_locs=[""] * n_proteins,
            is_poi=[True] * n_proteins,
        )
    )

    pros_df, pro_seqs_df = prep_v2_worker._step_2_create_pros_and_pro_seqs_dfs(
        pro_spec_df
    )

    peps_df, pep_seqs_df = prep_v2_worker._step_4_proteolysis(pro_seqs_df, "trypsin")

    return PrepV2Result(
        params=prep_params,
        _pros=pros_df,
        _pro_seqs=pro_seqs_df,
        _peps=peps_df,
        _pep_seqs=pep_seqs_df,
    )
