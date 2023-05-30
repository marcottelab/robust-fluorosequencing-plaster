"""Random Forest classifier."""
from pathlib import Path

import numpy as np

from plaster.run.rf_train_v2.rf_train_v2_result import RFTrainV2ResultDC
from plaster.run.rf_v2.rf_v2_params import RFV2Params
from plaster.run.rf_v2.rf_v2_result import RFV2ResultDC


def rf_classify(
    rf_v2_params: RFV2Params,
    rf_train_v2_result: RFTrainV2ResultDC,
    radmat: np.ndarray,  # Channel flattened
    folder: Path,
    *,
    result_class: RFV2ResultDC = RFV2ResultDC,
) -> RFV2ResultDC:
    assert len(radmat.shape) == 2

    assert isinstance(rf_train_v2_result, RFTrainV2ResultDC)
    classifier = rf_train_v2_result.classifier.get()

    winner_y, winner_scores, runnerup_y, runnerup_scores = classifier.classify(radmat)

    result = result_class(
        params=rf_v2_params,
        pred_pep_iz=winner_y,
        scores=winner_scores,
        runnerup_pep_iz=runnerup_y,
        runnerup_scores=runnerup_scores,
    )

    result.set_folder(folder)

    return result
