from dataclasses import dataclass

from dataclasses_json import DataClassJsonMixin

from plaster.run.base_result_dc import (
    BaseResultDC,
    LazyField,
    generate_flyte_result_class,
    lazy_field,
)
from plaster.run.rf_train_v2.rf_train_v2_params import RFTrainV2Params
from plaster.run.rf_train_v2.sklearn_rf import SKLearnRandomForestClassifier


@dataclass
class RFTrainV2ResultDC(BaseResultDC, DataClassJsonMixin):

    params: RFTrainV2Params
    classifier: LazyField = lazy_field(SKLearnRandomForestClassifier)

    def __repr__(self):
        return f"RFTrainV2ResultDC"


RFTrainV2FlyteResult = generate_flyte_result_class(RFTrainV2ResultDC)
