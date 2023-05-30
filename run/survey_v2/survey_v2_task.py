from flytekit import Resources, task

from plaster.run.prep.prep_result import PrepFlyteResult
from plaster.run.sim_v3.sim_v3_result import SimV3FlyteResult, SimV3Result
from plaster.run.survey_v2.survey_v2_params import SurveyV2Params
from plaster.run.survey_v2.survey_v2_result import SurveyV2FlyteResult, SurveyV2ResultDC
from plaster.run.survey_v2.survey_v2_worker import survey_v2

# TODO: I don't think this is currently used in the way surveys are done.


@task(requests=Resources(cpu="16", mem="16Gi"))
def survey_v2_flyte_task(
    survey_v2_params: SurveyV2Params,
    prep_result: PrepFlyteResult,
    sim_v3_result: SimV3FlyteResult,
) -> SurveyV2FlyteResult:
    result = survey_v2(
        survey_v2_params,
        prep_result.load_result(),
        sim_v3_result.load_result(),
        result_class=SurveyV2ResultDC,
    )

    return SurveyV2FlyteResult.from_inst(result)
