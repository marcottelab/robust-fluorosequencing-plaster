"""
Train the Random Forest classifier in poi_vs_all mode
"""
from pathlib import Path

import numpy as np
import structlog

from plaster.run.rf_train_v2.rf_train_v2_result import RFTrainV2ResultDC
from plaster.run.rf_train_v2.sklearn_rf import SKLearnRandomForestClassifier, poi_vs_all
from plaster.tools.utils import stats

logger = structlog.get_logger()


def _subsample(n_subsample, X, y):
    n_peps = np.max(y) + 1
    _y = []
    _X = []
    for pep_i in range(n_peps):
        args = np.argwhere(y == pep_i)
        arg_subsample = stats.subsample(args, n_subsample)
        _X += [X[arg_subsample]]
        _y += [y[arg_subsample]]

    X = np.vstack(_X)
    y = np.vstack(_y)
    return np.squeeze(X, axis=1), np.squeeze(y, axis=1)


def rf_train(
    rf_train_params,
    prep_result,
    sim_result,
    folder: Path,
    *,
    result_class=RFTrainV2ResultDC,
):
    log = logger.bind(
        n_samples_train=sim_result.params.n_samples_train,
        n_peps=prep_result.n_peps,
        n_pros=prep_result.n_pros,
    )

    X = sim_result.flat_train_radmat()

    if rf_train_params.classify_dyetracks:
        y = sim_result.train_true_dye_iz.arr()
    elif rf_train_params.use_poi_vs_all:
        y = poi_vs_all(sim_result.train_true_pep_iz, prep_result)
    else:
        y = sim_result.train_true_pep_iz

    if rf_train_params.n_subsample != -1:
        X, y = _subsample(rf_train_params.n_subsample, X, y)

    else:
        if sim_result.params.n_samples_train > 1000:
            log.warning(
                "RF classifier does not memory-scale well when n_samples_train > 1000"
            )

    if prep_result.n_peps > 1000 or (
        prep_result.n_pros > 100 and prep_result.n_pros != prep_result.n_peps
    ):
        log.warning(
            "RF classifier does not memory-scale well when n_peps > 1000 or n_pros > 100"
        )

    classifier = SKLearnRandomForestClassifier(rf_train_params.sklearn_rf_params)

    X_size = X.shape
    y_size = len(y)
    log.info("begin classifier.train", X_size=X_size, y_size=y_size)

    classifier.train(X, y)

    result = result_class(params=rf_train_params, classifier=classifier)

    result.set_folder(folder)

    return result
