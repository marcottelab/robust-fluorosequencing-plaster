from dataclasses import dataclass, field

MOVIE_HELP = """
Set to true when importing Nikon ND2 "movie" files.

In this mode, each .nd2 file is a collection of images taken sequentially for a single field.
This is in contrast to the typical mode where each .nd2 file is a chemical cycle spanning
all fields/channels.

Since all data for a given field is already in a single file, the parallel
scatter/gather employed by the "normal" ND2 import task is not necessary.

The "fields" from the .nd2 file become "cycles" as if the instrument had
taken 1 field with a lot of cycles.
"""

SELF_CALIB_HELP = """
If True, the calibration will be performed on the data itself.

Note that in order to get good results from self-calibration, the following should apply to your source data:
1. The density of peaks is in a sweet spot:
    - High enough that you get enough samples (this can somewhat be compensated with more fields)
    - Low enough that there aren't many collisions
2. There are enough fields such that we can get good sampling of the peaks in each region.
3. There are enough fields such that we can only use cycle 0 for the reg illum ballance.
"""

AA_LIST_HELP = """
List of amino acid labels corresponding to a labeling scheme.
Each entry corresponds to a single dye label. Multiple amino acids may be
labeled with the same label. For example, ['DE', 'K'] will place label_0
on D and E, and label_1 on K. Must contain at least one entry.
"""

PROTEINS_HELP = """
List of proteins involved in simulation.
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
List elements are dicts as follows:
    id : str
        The id of the protein being sequenced.
    seqstr : str
        The sequence of the protein in question.
    abundance : float (default = 1)
        The relative abundance of the protein in question. Defaults to 1.
    is_poi : bool (default = False)
        If True, the protein is a "protein of interest" that should have
        detailed statistics calculated during the eval step. If False, the
        protein is a distractor.
"""

ALLOW_EDMAN_CTERM_HELP = """
If True, allow Edman degradation of the last amino acid in the sequence. If
False, the last amino acid in the sequence will not be removed by Edman steps.
"""

CLASSIFIER_HELP = """
Which classifier to use.  Currently RF is the only option.
"""

DECOYS_HELP = """
Decoys are protein sequences which are expected to *not* be present in a
sample and are used to estimate the false discovery rate (ie. the rate at
which the classifier makes incorrect calls.) In cases where decoys are
helpful, this option will generate decoys automatically prior to
proteolysis. Available decoy options are:
    "none" (do not add decoys)
    "reverse" (add decoys that reverse the existing sequences)
    "shuffle" (shuffle the existing sequences)
"""

ERROR_SETTING_HELP = """
Dict of sim error settings. Entries are as follows:
    <error parameter name> : str
        The name of the error parameter being set. See gen/base_generator.py for
        a list of error parameters.
    <error parameter setting> : List[float]
        A list containing a single float that is the max likelihood estimate of
        the error parameter.
        TODO: More complex error parameters incorporating distributions.
Possible error settings (defaults in parentheses) include
    p_edman_failure (0.06)          : Edman miss
    p_detach (0.05)                 : Surface detach
    row_k_beta (1.0)                : Mean of row adjustment
    row_k_sigma (0.16)              : Stdev. of row adjustment
Channel specific settings (replace N with the channel number) include
    gain_mu.ch_N (7500)             : Brightness per dye
    gain_sigma.ch_N (1200)          : Stdev brightness for one dye
    bg_mu.ch_N (0)                  : Background brightness
    bg_sigma.ch_N (400)             : Std of zero count
    p_bleach.ch_N (0.05)            : Bleach rate per cycle
    p_non_fluorescent.ch_N (0.07)   : Dud rate
"""

FRET_INTERACTIONS_HELP = """
    Has no effect unless use_v3=True.
    List of FretInteraction dataclasses. Each dataclass has the following attributes
        donor    : Donor dye channel for this interaction.
        acceptor : Acceptor dye channel for this interaction.
        factor   : FRET factor for this interaction. The donor channel will be
                    multiplied by this factor.
        flat     : If True, apply the FRET factor to the donor once if there are
                    any acceptors. If False, apply the FRET factor once for each
                    acceptor so that the total factor is (fret ** n_acceptors).
                    Defaults to True.
    For example, a dataclass with donor: 0, acceptor: 1, fret: 0.9 signifies that
    channel 0 is a donor, channel 1 is an acceptor, and FRET reduces channel 0 to
    90% of its intensity.
"""

N_EDMANS_HELP = """
The number of Edman degradation cycles to simulate. Each Edman step removes
one amino acid from the N terminal (left end) of the protein string.
"""

N_MOCKS_HELP = """
The number of mock cycles to simulate. Mock cycles can trigger bleaching
and dye loss but do not result in any amino acid removal.
"""

N_PRES_HELP = """
The number of pre cycles to simulate. Pre cycles can trigger bleaching
and dye loss but do not result in any amino acid removal.
"""

N_PTMS_LIMIT_HELP = """
Limit on the number of post translational modifications simulated.
"""

N_SAMPLES_TRAIN_HELP = """
Number of training samples to use.
"""

N_SAMPLES_TEST_HELP = """
Number of test samples to use.
"""

PROTEASE_HELP = """
Protease treatment to apply to the entries in proteins. Creates peptide
fragments for further analysis. See entries in plaster/tools/aaseq/proteolyze.py
for options. Proteases may be combined in a treatment by joining them together
with "+", eg "lysc+endopro". Defaults to None, ie do not apply proteolysis.
"""

PROTEINS_OF_INTEREST_HELP = """
List of proteins of interest, corresponding to ids in the protein
argument. Either this must be supplied or is_poi key/value pairs must
be provided in the protein argument.
"""

RANDOM_SEED_HELP = """
Seed for the random number generator. Specify for reproducibility.
"""

USE_V3_HELP = """
Use _radmat_sim_v3, which can model FRET interactions but runs significantly
slower than the default. In development.
"""


@dataclass
class SimConfigHelp:
    MARKERS_HELP: str = field(
        default="""
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
        """
    )
    PROTEINS_HELP: str = field(
        default="""
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
        """
    )
    PROTEINS_CSV_HELP: str = field(
        default="""
        A path to a CSV file containing a list of proteins. Sequences are either
        retrieved from UniProt or included in the file.
        Accepted columns: Name, UniprotAC, Sequence, Abundance, POI, PTM.
        Required columns: UniprotAC or Sequence.
        Overrides the proteins field if both are specified.
        Loading of the sequences occurs during expand() of the SimConfig dataclass.
        Example file contents, which downloads protein sequences from UniProt:
            Name,UniprotAC,Abundance,POI
            PSMB10,P40306,502.0,1
            PSME3,P61289,83.0,1
        Example file contents, which contains protein sequences:
            Name,Sequence,Abundance,POI
            PSMB10,MLKPALEPRGGFSFEN,502.0,1
            PSME3,MASLLKVDQEVKLKVD,83.0,1
        """
    )
    SEQ_PARAMS_HELP: str = field(
        default="""
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
                Probabiltiy that an Edman cycle will fail to detach an amino acid. (Between 0 and 1.)
            row_k_sigma : float
                Row-to-row standard deviation for all signal gains. Expressed as a fraction of the
                signal gain. (Greater than 0.)
        """
    )
    FRETS_HELP: str = field(
        default="""
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
        """
    )
    PROTEASES_HELP: str = field(
        default="""
        List of protease treatments to apply to the entries in proteins. Creates peptide
        fragments for further analysis. See entries in plaster/tools/aaseq/proteolyze.py
        for options.
        """
    )
    DECOY_MODE_HELP: str = field(
        default="""
        Decoys are protein sequences which are expected to *not* be present in a
        sample and are used to estimate the false discovery rate (ie. the rate at
        which the classifier makes incorrect calls.) In cases where decoys are
        helpful, this option will generate decoys automatically prior to
        proteolysis. Available decoy options are:
            - "none" (do not add decoys, same as default)
            - "reverse" (add decoys that reverse the existing sequences)
            - "shuffle" (shuffle the existing sequences)
        """
    )
    SHUFFLE_N_HELP: str = field(
        default="""
        Generate N-many shuffles of the input peptides. Only applicable when
        decoy_mode="shuffle".
        """
    )
    N_SAMPLES_TRAIN_HELP: str = field(default="Number of training samples to use.")
    N_SAMPLES_TEST_HELP: str = field(default="Number of test samples to use.")
    RANDOM_SEED_HELP: str = field(
        default="""
        (Not Yet Implemented) Seed for the random number generator. Specify for
        reproducibility in testing.
        """
    )
    IS_PHOTOBLEACHING_RUN_HELP: str = field(
        default="""
        If True, this is a photobleaching run, and different sim calculations will be
        performed.
        """
    )
    PHOTOBLEACHING_N_CYCLES_HELP: str = field(
        default="""
        Number of photobleaching cycles to perform. Only relevant if
        is_photobleaching_run = True.
        """
    )
    PHOTOBLEACHING_RUN_N_DYE_COUNT_HELP: str = field(
        default="""
        Number of photobleaching dyes. Only relevant if is_photobleaching_run = True.
        """
    )


@dataclass
class VFSConfigHelp:
    CLASSIFIER_HELP: str = field(
        default="""
        Select a classification method. Current options are;
            - Classifier.RF (random forest).
        """
    )
