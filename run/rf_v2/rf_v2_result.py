from dataclasses import dataclass

import numpy as np
import pandas as pd
from dataclasses_json import DataClassJsonMixin

from plaster.run.base_result_dc import (
    BaseResultDC,
    LazyField,
    generate_flyte_result_class,
    lazy_field,
)
from plaster.run.rf_v2.rf_v2_params import RFV2Params


@dataclass
class RFV2ResultDC(BaseResultDC, DataClassJsonMixin):
    params: RFV2Params
    true_pep_iz: LazyField = lazy_field(np.ndarray)
    pred_pep_iz: LazyField = lazy_field(np.ndarray)
    scores: LazyField = lazy_field(np.ndarray)
    runnerup_pep_iz: LazyField = lazy_field(np.ndarray)
    runnerup_scores: LazyField = lazy_field(np.ndarray)
    train_pep_recalls: LazyField = lazy_field(np.ndarray)

    def __repr__(self):
        try:
            return f"RFV2ResultDC with average score {np.mean(self.scores.get())}"
        except:
            return "RFV2ResultDC"

    def rf_sim_df(self):
        return pd.DataFrame.from_dict(
            {
                x: self.__getattribute__(x).get()
                for x in [
                    "true_pep_iz",
                    "pred_pep_iz",
                    "runnerup_pep_iz",
                    "scores",
                    "runnerup_scores",
                ]
            },
            orient="columns",
        )

    def rf_sim_eval(self, score_cutoff: float = 0.0, try_runnerup: bool = True):
        work_df = self.rf_sim_df()
        pep_recalls = self.train_pep_recalls.get()

        if try_runnerup:
            work_df.scores = np.where(
                work_df.pred_pep_iz != 0, work_df.scores, work_df.runnerup_scores
            )
            work_df.pred_pep_iz = np.where(
                work_df.pred_pep_iz != 0, work_df.pred_pep_iz, work_df.runnerup_pep_iz
            )
        # only investigate the values that we get hits from even if we check runnerups
        hit_idxs = np.unique(work_df.pred_pep_iz[work_df.pred_pep_iz != 0])
        buffer = {}
        for hit_idx in hit_idxs:
            pred_df = work_df.loc[
                np.logical_and(
                    work_df.pred_pep_iz == hit_idx, work_df.scores >= score_cutoff
                ),
                :,
            ]
            true_df = work_df.loc[work_df.true_pep_iz == hit_idx, :]
            tp_df = pred_df.loc[pred_df.pred_pep_iz == pred_df.true_pep_iz, :]
            fp_df = pred_df.loc[pred_df.pred_pep_iz != pred_df.true_pep_iz, :]
            fn_df = true_df.loc[true_df.pred_pep_iz != true_df.true_pep_iz, :]
            n_tp = tp_df.shape[0]
            n_fp = fp_df.shape[0]
            n_fn = fn_df.shape[0]
            n_fn_extra = int(np.round(true_df.shape[0] * (1.0 - pep_recalls[hit_idx])))
            n_fn_corr = n_fn + n_fn_extra
            precision = 0.0
            recall = 0.0
            recall_corr = 0.0
            if n_tp > 0.0:
                precision = n_tp / (n_tp + n_fp)
                recall = n_tp / (n_tp + n_fn)
                recall_corr = n_tp / (n_tp + n_fn_corr)
            min_tp_score = np.min(tp_df.scores)
            max_tp_score = np.max(tp_df.scores)
            min_fp_score = np.min(fp_df.scores)
            max_fp_score = np.max(fp_df.scores)
            buffer[hit_idx] = {
                "n_tp": n_tp,
                "n_fp": n_fp,
                "n_fn": n_fn,
                "n_fn_corr": n_fn_corr,
                "precision": precision,
                "recall": recall,
                "recall_corr": recall_corr,
                "min_tp": min_tp_score,
                "max_tp": max_tp_score,
                "min_fp": min_fp_score,
                "max_fp": max_fp_score,
            }

        return pd.DataFrame.from_dict(buffer, orient="index").sort_index().fillna(0.0)


RFV2FlyteResult = generate_flyte_result_class(RFV2ResultDC)
