from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from dataclasses_json import DataClassJsonMixin

from plaster.tools.aaseq import proteolyze


@dataclass
class Protein(DataClassJsonMixin):
    """
    Protein sequencing information.

    Attributes
    ----------
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
        If specified, indicates post-translational modification locations. Specify an
        empty string or None to indicate no post-translational modifications.
    """

    name: str
    sequence: str
    is_poi: bool = True
    abundance: float = 1.0
    ptm_locs: Optional[str] = ""

    def __post_init__(self):
        if self.abundance:
            self.abundance = float(self.abundance)
            if self.abundance < 0.0:
                raise ValueError("Protein abundance must be >= 0")
        else:
            self.abundance = 0.0
        if not self.ptm_locs:
            self.ptm_locs = ""
        if not self.name:
            raise ValueError("Protein name must contain at least 1 character")


@dataclass
class PrepV2Params(DataClassJsonMixin):

    _PHOTOBLEACHING_PSEUDO_AA = "X"

    proteins: List[Protein] = field(default_factory=list)
    proteases: List[str] = field(default_factory=list)
    decoy_mode: Optional[str] = None
    shuffle_n: int = 1
    is_photobleaching_run: bool = False
    photobleaching_n_cycles: int = 1
    photobleaching_run_n_dye_count: int = 1
    # deprecated below this line
    # drop_duplicates: bool = False
    # n_peps_limit: int = -1
    # n_ptms_limit: int = -1

    def __post_init__(self):
        if self.shuffle_n < 1:
            raise ValueError("shuffle_n must be >= 1")
        protease_test = set(self.proteases) - set(proteolyze.proteases.keys())
        if len(protease_test) > 0:
            raise ValueError(
                f"Supplied proteases {protease_test} not found in proteolyze.proteases"
            )

        self.protein_df = pd.DataFrame(self.proteins)
