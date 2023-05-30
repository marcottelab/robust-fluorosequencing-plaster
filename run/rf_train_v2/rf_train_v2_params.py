from dataclasses import dataclass, fields
from typing import Optional

from dataclasses_json import DataClassJsonMixin

from plaster.tools.utils import utils


@dataclass
class SKLearnRFParams(utils.DataclassUnpickleFromMunchMixin, DataClassJsonMixin):
    # these are SKLearn classifier settings
    # mirroring defaults for sklearn classifier except where noted

    # This is the number of trees, and presumably this was turned down to help
    # with memory scaling.  But it may produce better results to have more
    # shallow trees.  See max_depth.
    n_estimators: int = 100  # this is default, dsw used 10

    criterion: str = "gini"

    # This is a good candidate for dealing with memory scaling.
    # Look at the current depth of existing trees and set this
    # to some fraction of that.
    max_depth: Optional[int] = None  # default

    min_samples_split: int = 2

    # This is probably important, and this value seems arbitrary. In particular,
    # it can't really stay the same if you're changing the size of the training set.
    # So it may be smarter, if you're going to set some count > 1, to use a fraction
    # which is interpreted as fraction of training samples.
    # min_samples_leaf: int = 50  # dsw
    min_samples_leaf = 1  # default

    min_weight_fraction_leaf: float = 0.0
    max_features: Optional[str] = "sqrt"
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    oob_score: bool = False

    # only applies during training
    # during classification zap is used and we manually turn this down to 1 and
    # parallel with zap.
    n_jobs: int = -1  # this means use all available cores

    random_state: Optional[int] = None
    verbose: int = 0
    warm_start: bool = False
    class_weight: Optional[str] = None
    ccp_alpha: float = 0.0
    max_samples: Optional[int] = None


@dataclass
class RFTrainV2Params(utils.DataclassUnpickleFromMunchMixin, DataClassJsonMixin):

    # this holds the SKLearnRFParams
    sklearn_rf_params: SKLearnRFParams = SKLearnRFParams()

    # these are our settings
    n_subsample: int = -1
    use_poi_vs_all: bool = False
    classify_dyetracks: bool = False

    def __contains__(self, elem):
        return elem in [field.name for field in fields(self)]
