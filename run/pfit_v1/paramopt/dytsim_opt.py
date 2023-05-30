import ctypes

import matplotlib.pyplot as plt
import numpy as np

import plaster.genv2.generators.sim as simgen
import plaster.run.sim_v3.c_dytsim.dytsim as dytsim
from plaster.run.prep_v2.prep_v2_worker import prep_v2
from plaster.run.sim_v3 import dyt_helpers
from plaster.tools.c_common import c_common_tools
from plaster.tools.utils.utils import timing

from . import helpers


def config_plaster(config, dyetides):
    protein_list = helpers.gen_protein_list(dyetides, 0)

    markers = []
    for i, (aa, p_bleach, p_dud) in enumerate(
        zip(config["marker_labels"], config["p_bleach"], config["p_dud"])
    ):
        markers += [
            simgen.Marker(
                aa=aa,
                p_bleach=p_bleach,
                p_dud=p_dud,
            )
        ]

    trial_config = simgen.SimConfig(
        type="sim",
        job="/dev/null",
        markers=markers,
        seq_params=simgen.SeqParams(
            n_edmans=config["n_edmans"],
            p_edman_failure=config["p_edman_failure"],
            p_label_failure=config["p_label_failure"],
            p_cyclic_block=config["p_cyclic_block"],
            p_initial_block=config["p_initial_block"],
            p_detach=config["p_detach"],
            allow_edman_cterm=config["allow_edman_cterm"],
        ),
        proteins=protein_list,
        seed=0,
    )

    prep_params = simgen.generate_prep_params(trial_config)
    sim_params = simgen.generate_sim_params(trial_config)
    prep_result = prep_v2(prep_params=prep_params, folder="/dev/null")
    synth_pcbs = sim_params.pcbs(prep_result.pepseqs__with_decoys())
    return sim_params, synth_pcbs
