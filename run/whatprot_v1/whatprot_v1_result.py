import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from dataclasses_json import DataClassJsonMixin

from plaster.run.base_result_dc import (
    BaseResultDC,
    LazyField,
    generate_flyte_result_class,
    lazy_field,
)
from plaster.run.sim_v3.sim_v3_params import SimV3Params
from plaster.run.whatprot_v1.whatprot_v1_params import WhatprotV1Params


@dataclass
class WhatprotPreV1ResultDC(BaseResultDC, DataClassJsonMixin):

    params: WhatprotV1Params
    sim_train_params: SimV3Params  # source of params input to whatprot

    # I want to have sim_train_params and optionally sim_test_params as well
    # so that it is easy to see in your Whatprot results (whether classify or
    # fitting) the params used for "training" (or initial params in fitting)
    # and the params used to generate the data that is classified or fit.

    # wp_seq_params: dict                  # the params input to whatprot

    test_data_type: str  # sim or sigproc
    sim_test_params: Optional[SimV3Params] = None  # params used to gen test data.

    # The Pre task writes a bunch of files to the folder that are
    # used as input to whatprot in the Run task.  These files will
    # get copied to the wp_pre task folder inside of your jobs_folder.
    # If you want those paths discoverable in the result.yaml, you need
    # to put FlyteFile references here in this result class.

    def __repr__(self):
        return "WhatprotPreV1ResultDC"


WhatprotPreV1FlyteResult = generate_flyte_result_class(WhatprotPreV1ResultDC)


@dataclass
class WhatprotClassifyV1ResultDC(BaseResultDC, DataClassJsonMixin):
    params: WhatprotV1Params
    cmd_list: list[str]  # the whatprot command line used to execute

    # The Run task also writes a file to its working_directory (which
    # will be copied to the wp_run task subfolder of your jobs_folder).
    # This file is read by the next task.  See comment in Pre task.

    def __repr__(self):
        return "WhatprotClassifyV1ResultDC"


WhatprotClassifyV1FlyteResult = generate_flyte_result_class(WhatprotClassifyV1ResultDC)


@dataclass
class WhatprotFitV1ResultDC(BaseResultDC, DataClassJsonMixin):
    params: WhatprotV1Params
    cmd_list: list[str]  # the whatprot command line used to execute

    params_fit: dict  # a dictionary of param => best_fit_value

    # The Run task also writes a file to its working_directory (which
    # will be copied to the wp_run task subfolder of your jobs_folder).
    # This file is read by the next task.  See comment in Pre task.

    def __repr__(self):
        return "WhatprotFitV1ResultDC"


WhatprotFitV1FlyteResult = generate_flyte_result_class(WhatprotFitV1ResultDC)


@dataclass
class WhatprotFitBootstrapV1ResultDC(BaseResultDC, DataClassJsonMixin):
    params: WhatprotV1Params
    cmd_list: list[str]  # the whatprot command line used to execute

    bootstrap_bounds_lower: dict  # param -> CI lower bound
    bootstrap_bounds_upper: dict  # param -> CI upper bound
    bootstrap_csv: str  # the csv file containing all bootstrap fits

    # The Run task also writes a file to its working_directory (which
    # will be copied to the wp_run task subfolder of your jobs_folder).
    # This file is read by the next task.  See comment in Pre task.

    def __repr__(self):
        return "WhatprotFitBootstrapV1ResultDC"

    def bootstrap_df(self):
        return pd.read_csv(self.bootstrap_csv)


WhatprotFitBootstrapV1FlyteResult = generate_flyte_result_class(
    WhatprotFitBootstrapV1ResultDC
)


@dataclass
class WhatprotPostClassifyV1ResultDC(BaseResultDC, DataClassJsonMixin):
    params: WhatprotV1Params
    true_pep_iz: LazyField = lazy_field(np.ndarray)
    pred_pep_iz: LazyField = lazy_field(np.ndarray)
    scores: LazyField = lazy_field(np.ndarray)
    train_pep_recalls: LazyField = lazy_field(np.ndarray)

    def __repr__(self):
        try:
            return f"WhatprotPostClassifyV1ResultDC with average score {np.mean(self.scores.get())}"
        except:
            return "WhatprotPostClassifyV1ResultDC"

    def wp_sim_df(self):
        return pd.DataFrame.from_dict(
            {
                x: self.__getattribute__(x).get()
                for x in [
                    "true_pep_iz",
                    "pred_pep_iz",
                    #                    "runnerup_pep_iz",  #  these are avail for erisyon RF classifiers but not whatprot at this time.
                    "scores",
                    #                    "runnerup_scores",
                ]
            },
            orient="columns",
        )


WhatprotPostClassifyV1FlyteResult = generate_flyte_result_class(
    WhatprotPostClassifyV1ResultDC
)
