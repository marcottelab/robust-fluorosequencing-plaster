from dataclasses import dataclass

import numpy as np
from dataclasses_json import DataClassJsonMixin
from munch import Munch

import plaster.tools.utils.utils as utils
from plaster.run.rf_train_v2.rf_train_v2_params import SKLearnRFParams
from plaster.tools.zap import zap


def poi_vs_all(true_pep_iz, prep_result):
    """
    true_pep_iz refers to the full set of all peptides.

    This function returns remapped values of true_pep_iz such that
    each peptide that is from a POI keeps its true_pep_i
    whereas all other peptides are remapped to class 0, ie "OTHER"
    """

    df = prep_result.peps__pois()
    lut = df.pep_i.values
    other_mask = df.is_poi.values != 1
    lut[other_mask] = 0
    return lut[true_pep_iz]


def _do_predict(classifier, X):
    """
    The Scikit Learn Random Classifier has a predict() and a predict_proba()
    functions. But the expensive part is the predict_proba().
    Oddly, the predict() calls predict_proba() so I was previously doing
    the expensive part twice. Since I only want a single call per row
    I need to re-implement the predict() which
    is nothing more than taking the best score on each row.

    I also extract both the winner (best score) and the runnerup
    under the theory that the distance between these might be informative.

    Note, this must be a module-level function so that it can pickle.

    n_jobs is set to 1 for classification because this function is
    called by individual zap job workers.
    """

    assert len(X.shape) == 2

    # set n_jobs to 1 for classification for individual zap workers
    classifier.n_jobs = 1

    # all_scores is the score for each row of X against EVERY classification class
    all_scores = classifier.predict_proba(X)

    # Sort by score along each row so that we can get the winner and runner up
    sorted_iz = np.argsort(all_scores, axis=1)
    winner_iz = sorted_iz[:, -1]
    runnerup_iz = sorted_iz[:, -2]

    winner_scores = all_scores[np.arange(all_scores.shape[0]), winner_iz]
    runnerup_scores = all_scores[np.arange(all_scores.shape[0]), runnerup_iz]

    winner_y = classifier.classes_.take(winner_iz, axis=0)
    runnerup_y = classifier.classes_.take(runnerup_iz, axis=0)

    return winner_y, winner_scores, runnerup_y, runnerup_scores


@dataclass
class SKLearnRandomForestClassifier(DataClassJsonMixin):

    classifier_args: SKLearnRFParams

    def __post_init__(self):
        from sklearn.ensemble import RandomForestClassifier  # Defer slow import

        self.n_progress_jobs = self.classifier_args.n_estimators
        self.classifier = RandomForestClassifier(**(self.classifier_args.to_dict()))

    def train(self, X, y):
        self.classifier.fit(X, y)

    def info(self):
        return Munch(
            feature_importances=self.classifier.feature_importances_,
            n_outputs=self.classifier.n_outputs_,
            n_features=self.classifier.n_features_,
            n_classes=self.classifier.n_classes_,
        )

    def classify(self, X):
        assert len(X.shape) == 2

        n_rows = X.shape[0]

        if n_rows < 100:
            winner_y, winner_scores, runnerup_y, runnerup_scores = _do_predict(
                classifier=self.classifier, X=X
            )
        else:
            n_work_orders = n_rows // 100

            with zap.Context(progress=None, trap_exceptions=False):
                results = zap.work_orders(
                    [
                        Munch(classifier=self.classifier, X=X, fn=_do_predict)
                        for X in np.array_split(X, n_work_orders, axis=0)
                    ]
                )
            winner_y = utils.listi(results, 0)
            winner_scores = utils.listi(results, 1)
            runnerup_y = utils.listi(results, 2)
            runnerup_scores = utils.listi(results, 3)

            winner_y = np.concatenate(winner_y)
            winner_scores = np.concatenate(winner_scores)
            runnerup_y = np.concatenate(runnerup_y)
            runnerup_scores = np.concatenate(runnerup_scores)

        return winner_y, winner_scores, runnerup_y, runnerup_scores
