"""Sim config for generators."""

import typing as t
from dataclasses import dataclass
from pathlib import Path

import structlog
import yaml
from dataclasses_json import DataClassJsonMixin
from flytekit import task, workflow

import plaster.genv2.help_texts as help_texts
from plaster.genv2 import gen_config, gen_utils
from plaster.run.prep_v2.prep_v2_params import PrepV2Params, Protein
from plaster.run.prep_v2.prep_v2_result import PrepV2FlyteResult
from plaster.run.prep_v2.prep_v2_task import prep_flyte_task
from plaster.run.sim_v3.sim_v3_params import Fret, Marker, SeqParams, SimV3Params
from plaster.run.sim_v3.sim_v3_result import SimV3FlyteResult
from plaster.run.sim_v3.sim_v3_task import (
    sim_v3_big_flyte_task,
    sim_v3_flyte_task,
    sim_v3_xlarge_flyte_task,
)
from plaster.tools.flyte import task_configs

logger = structlog.get_logger()


class SimConfigException(Exception):
    def __init__(self, message=None):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message


SimConfigHelp = help_texts.SimConfigHelp()


@dataclass
class SimConfig(gen_config.BaseGenConfig, DataClassJsonMixin):
    """Configuration dataclass for simulation runs.

    Attributes
    ----------
    type: str (inherited)
        Specifies the simulation type. Current options are:
            - "vfs"
    job: str
        Specifies the path used to save sim outputs.
    save: bool (default=True)
        If True, save sim outputs to the path described in the job field.
    force: bool (default=True)
        If True, overwrite existing sim outputs when saving.
    markers : list[Marker]
        Labeling scheme for this simulation.
        Marker documentation:
            aa : str
                Amino acid[s] to which the markers are attached. One or more uppercase
                letters, eg "C" or "DE". If multiple letters are specified, then
                each letter will receive the same marker.
            gain_mu : float
                Signal mean for a single instance of this marker. (Greater than 0.)
            gain_sigma : float
                Signal standard deviation for a single instance of this marker. (Greater than 0.)
            bg_mu : float
                Background mean for this marker channel.
            bg_sigma : float
                Background standard deviation for this marker channel. (Greater than 0.)
            p_bleach : float
                Per-cycle bleaching probability for this marker. (Between 0. and 1.)
            p_dud : float
                Dud dye probability for this marker. (Between 0. and 1.)
    proteins : list[Protein]
        Proteins involved in this simulation.
        Protein documentation:
            name: str
                Name or label of the protein.
            sequence: str
                Protein and peptide sequences are specified in IUPAC; N to C order.
                (http://publications.iupac.org/pac/1984/pdf/5605x0595.pdf)
                Special rules:
                    * Whitespace is ignored
                        "AB CD" = "ABCD"
                    * "." can be used in place of "X"
                        "AB..CD" = "ABXXCD"
                    * Anything wrapped in () is dropped.
                        "AB(a comment)CD" = "ABCD"
                    * Square brackets are modifications of the previous amino-acid,
                    usually used to indicate a Post-Translational-Modification (PTM)
                        "AS[p]D" = "A" + "S[p]" + "D"
                    * Curly brackets are reserved for future use
            is_poi: bool
                Indicates a protein of interest. If True, this protein and its associated
                peptides will be tracked in detail throughout the simulation process. If False,
                they will be treated as distractors and detailed analysis results will not be
                shown.
            abundance: float
                Relative abundance of the protein.
            ptm_locs: Optional[str]
                If specified, indicates post translational modification locations.
    seq_params : SeqParams
        Sequencing parameters for this simulation.
        SeqParams documentation:
            n_pre : int
                Number of preliminary illumination cycles used for image calibration.
            n_mocks : int
                Number of chemical wash cycles done without Edman degradation.
            n_edmans : int
                Number of Edman degradation cycles.
            allow_edman_cterm : bool
                Allow Edman degradation of the C-terminal amino acid. If False, the C-terminal
                amino acid will remain regardless of the number of Edman degradation steps.
            p_detach : float
                Probability of the peptide detaching per cycle. (Between 0 and 1.)
            p_edman_failure : float
                Probability that an Edman cycle will fail to detach an amino acid. (Between 0 and 1.)
            row_k_sigma : float
                Row-to-row standard deviation for all signal gains. Expressed as a fraction of the
                signal gain. (Greater than 0.)
    frets : list[Fret]
        FRET interactions for this simulation.
        Fret documentation:
            donor: int
                The donor dye channel. This channel loses intensity as a result of
                interactions with the acceptor channel.
            acceptor: int
                The acceptor dye channel. This channel is currently unaffected by
                FRET, since we do not currently observe acceptors during donor
                illumination.
            fret_mu: float
                Mean FRET efficiency of the donor-acceptor pair. For a single
                donor-acceptor pair, intensity will be modified by
                (1.0 - rng.normal(fret_mu, fret_sigma)). Values will be limited to
                the range 0.0-1.0.
            fret_sigma: float
                Standard deviation of the FRET efficiency of the donor-acceptor pair.
            flat: bool (default=True)
                If True, the FRET efficiency will be applied to donors once if any
                acceptors or quenching donors are present. If False, the FRET
                efficiency will be applied once per acceptor or quenching donor, and
                the total FRET effect will be = (1.0 - fret) ** n_acceptors.
    proteases : list[str]
        List of protease treatments to apply to the entries in proteins. Creates peptide
        fragments for further analysis. See entries in plaster/tools/aaseq/proteolyze.py
        for options.
    classifier : Classifier (default=Classifier.RF)
        Select a classification method. Current options are;
            - Classifier.RF (random forest).
    decoy_mode : Optional[str]
        Decoys are protein sequences which are expected to *not* be present in a
        sample and are used to estimate the false discovery rate (ie. the rate at
        which the classifier makes incorrect calls.) In cases where decoys are
        helpful, this option will generate decoys automatically prior to
        proteolysis. Available decoy options are:
            - "none" (do not add decoys, same as default)
            - "reverse" (add decoys that reverse the existing sequences)
            - "shuffle" (shuffle the existing sequences)
    shuffle_n : int (default=1)
        Generate N-many shuffles of the input peptides. Only applicable when
        decoy_mode="shuffle".
    n_samples_train : int (default=5000)
        Number of training samples to use.
    n_samples_test : int (default=1000)
        Number of test samples to use.
    random_seed: int (default=-1 (do not seed the generator))
        (Not Yet Implemented) Seed for the random number generator. Specify for
        reproducibility in testing.
    is_photobleaching_run: bool (default=False)
        If True, this is a photobleaching run, and different sim calculations will be
        performed.
    photobleaching_n_cycles: int (default=1)
        Number of photobleaching cycles to perform. Only relevant if
        is_photobleaching_run = True.
    photobleaching_run_n_dye_count: int (default=1)
        Number of photobleaching dyes. Only relevant if is_photobleaching_run = True.
    """

    markers: list[Marker] = gen_config.gen_field(
        help=SimConfigHelp.MARKERS_HELP, default_factory=list
    )
    proteins: list[Protein] = gen_config.gen_field(
        help=SimConfigHelp.PROTEINS_HELP, default_factory=list
    )
    proteins_csv: str = gen_config.gen_field(
        help=SimConfigHelp.PROTEINS_CSV_HELP, default_factory=list
    )
    seq_params: SeqParams = gen_config.gen_field(
        help=SimConfigHelp.SEQ_PARAMS_HELP, default_factory=SeqParams
    )
    frets: list[Fret] = gen_config.gen_field(
        help=SimConfigHelp.FRETS_HELP, default_factory=list
    )
    proteases: list[str] = gen_config.gen_field(
        help=SimConfigHelp.PROTEASES_HELP, default_factory=list
    )
    n_samples_train: int = gen_config.gen_field(
        help=SimConfigHelp.N_SAMPLES_TRAIN_HELP, default=5000
    )
    n_samples_test: int = gen_config.gen_field(
        help=SimConfigHelp.N_SAMPLES_TEST_HELP, default=1000
    )
    decoy_mode: t.Optional[str] = gen_config.gen_field(
        help=SimConfigHelp.DECOY_MODE_HELP, default=None
    )
    shuffle_n: int = gen_config.gen_field(help=SimConfigHelp.SHUFFLE_N_HELP, default=1)
    # TODO: Propagate this forward into sim_params
    random_seed: int = gen_config.gen_field(
        help=SimConfigHelp.RANDOM_SEED_HELP, default=-1
    )
    is_photobleaching_run: bool = gen_config.gen_field(
        help=SimConfigHelp.IS_PHOTOBLEACHING_RUN_HELP, default=False
    )
    photobleaching_n_cycles: int = gen_config.gen_field(
        help=SimConfigHelp.PHOTOBLEACHING_N_CYCLES_HELP, default=1
    )
    photobleaching_run_n_dye_count: int = gen_config.gen_field(
        help=SimConfigHelp.PHOTOBLEACHING_RUN_N_DYE_COUNT_HELP, default=1
    )
    force: bool = False
    span_dyetracks: bool = False
    n_span_labels_list: list[int] = gen_config.gen_field(
        help="list indicating how many labels there are in each channel",
        default_factory=list,
    )
    test_decoys: bool = False
    seed: t.Optional[int] = None

    def validate(self):
        if not self.proteins and not self.proteins_csv:
            return gen_config.ValidationResult(
                valid=False,
                message=ValueError("Either proteins or proteins_csv must be specified"),
            )
        return gen_config.ValidationResult(True)

    def fetch_protein_sequences(self):
        if self.proteins_csv:
            import plaster.tools.aaseq.protein as protein
            from plaster.genv2.gen_utils import resolve_job_folder

            fn = resolve_job_folder(self.proteins_csv)

            try:
                proteins_dc = protein._build_protein_list(fn)[0]
                logger.info(f"Merging protein CSV into config", proteins_csv=fn)
            except OSError:
                logger.info(f"Unable to load protein CSV into config", proteins_csv=fn)
                raise SimConfigException(f"Unable to load protein CSV file {fn}")
            else:
                self.proteins = proteins_dc
        if not self.proteins:
            raise SimConfigException("proteins must contain at least one entry")


def generate_prep_params(
    config: SimConfig,
) -> PrepV2Params:
    return PrepV2Params(
        proteins=config.proteins,
        proteases=config.proteases,
        decoy_mode=config.decoy_mode,
        shuffle_n=config.shuffle_n,
        is_photobleaching_run=config.is_photobleaching_run,
        photobleaching_n_cycles=config.photobleaching_n_cycles,
        photobleaching_run_n_dye_count=config.photobleaching_run_n_dye_count,
    )


def generate_sim_params(
    config: SimConfig,
) -> SimV3Params:
    return SimV3Params(
        markers=config.markers,
        frets=config.frets,
        seq_params=config.seq_params,
        n_samples_train=config.n_samples_train,
        n_samples_test=config.n_samples_test,
        is_survey=False,
        span_dyetracks=config.span_dyetracks,
        n_span_labels_list=config.n_span_labels_list,
        test_decoys=config.test_decoys,
        seed=config.seed,
    )


@task
def extract_params(config: SimConfig) -> tuple[PrepV2Params, SimV3Params]:
    prep_params = generate_prep_params(config)
    sim_params = generate_sim_params(config)
    return prep_params, sim_params


@task(task_config=task_configs.generate_efs_task_config())
def write_sim_only_job_folder_result(
    config: SimConfig,
    prep_flyte_result: PrepV2FlyteResult,
    sim_flyte_result: SimV3FlyteResult,
) -> None:

    # Write job folder if specified
    if config.job:
        job_path = gen_utils.write_job_folder(
            job_folder=config.job, config_dict=config.to_dict()
        )

        # Write result folders
        prep_flyte_result.save_to_disk(job_path / "prep")
        sim_flyte_result.save_to_disk(job_path / "sim")


def load_sim_only_results(
    path: str,
) -> tuple[SimConfig, PrepV2FlyteResult, SimV3FlyteResult]:
    """
    Load sim results from a given path.

    Args
    ----
    path: str
        Path to a sim results folder.
    """
    path = Path(path).expanduser()
    sim_config = SimConfig.from_dict(
        yaml.safe_load((path / "job_manifest.yaml").read_text())["config"]
    )
    prep_flyte_result = PrepV2FlyteResult.load_from_disk(path / "prep")
    sim_flyte_result = SimV3FlyteResult.load_from_disk(path / "sim")

    return (
        sim_config,
        prep_flyte_result,
        sim_flyte_result,
    )


@workflow
def sim_only_workflow(
    config: SimConfig,
) -> tuple[PrepV2FlyteResult, SimV3FlyteResult]:
    """
    Sim generation workflow.

    Args
    ----
    config : SimConfig
        Specifies VFS simulation configuration.

    Returns
    -------
    Tuple of Result objects containing the results of the sim run.

    """

    prep_params, sim_params = extract_params(config=config)
    prep_flyte_result = prep_flyte_task(prep_params=prep_params)
    sim_flyte_result = sim_v3_flyte_task(
        sim_params=sim_params, prep_flyte_result=prep_flyte_result
    )

    write_sim_only_job_folder_result(
        config=config,
        prep_flyte_result=prep_flyte_result,
        sim_flyte_result=sim_flyte_result,
    )

    return (
        prep_flyte_result,
        sim_flyte_result,
    )


@workflow
def sim_big_workflow(
    config: SimConfig,
) -> tuple[PrepV2FlyteResult, SimV3FlyteResult]:
    """
    Sim generation workflow.

    Args
    ----
    config : SimConfig
        Specifies VFS simulation configuration.

    Returns
    -------
    Tuple of Result objects containing the results of the sim run.

    """

    prep_params, sim_params = extract_params(config=config)
    prep_flyte_result = prep_flyte_task(prep_params=prep_params)
    sim_flyte_result = sim_v3_big_flyte_task(
        sim_params=sim_params, prep_flyte_result=prep_flyte_result
    )

    write_sim_only_job_folder_result(
        config=config,
        prep_flyte_result=prep_flyte_result,
        sim_flyte_result=sim_flyte_result,
    )

    return (
        prep_flyte_result,
        sim_flyte_result,
    )


@workflow
def sim_xlarge_workflow(
    config: SimConfig,
) -> tuple[PrepV2FlyteResult, SimV3FlyteResult]:
    """
    Sim generation workflow.

    Args
    ----
    config : SimConfig
        Specifies VFS simulation configuration.

    Returns
    -------
    Tuple of Result objects containing the results of the sim run.

    """

    prep_params, sim_params = extract_params(config=config)
    prep_flyte_result = prep_flyte_task(prep_params=prep_params)
    sim_flyte_result = sim_v3_xlarge_flyte_task(
        sim_params=sim_params, prep_flyte_result=prep_flyte_result
    )

    write_sim_only_job_folder_result(
        config=config,
        prep_flyte_result=prep_flyte_result,
        sim_flyte_result=sim_flyte_result,
    )

    return (
        prep_flyte_result,
        sim_flyte_result,
    )


def generate(config: SimConfig) -> gen_config.GenerateResult:
    """Generate a Sim job from a SimConfig dataclass.

    Args
    ----
    config : SimConfig
        A SimConfig specifying sim+classify configuration

    Returns
    -------
    A gen_config.GenerateResult object describing the runs and static_reports to be executed
    for this Sim job.
    """

    # Flyte doesn't currently have a concept of runs within a job, but this may change
    # when this generate fn takes some kind of permutation information to help users
    # create a series of related jobs (formerly called runs within a job).  Or similar.
    runs = []

    static_reports = []

    config.fetch_protein_sequences()

    return gen_config.GenerateResult(runs=runs, static_reports=static_reports)


generator = gen_config.Generator(SimConfig, generate, workflow=sim_only_workflow)
generator_big = gen_config.Generator(SimConfig, generate, workflow=sim_big_workflow)
generator_xl = gen_config.Generator(SimConfig, generate, workflow=sim_xlarge_workflow)
