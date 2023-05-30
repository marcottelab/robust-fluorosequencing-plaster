import itertools
from dataclasses import dataclass
from typing import Optional

from dataclasses_json import DataClassJsonMixin

import plaster.genv2.gen_config as gen_config
from plaster.tools.aaseq.proteolyze import proteases


@dataclass
class ProteaseLabelScheme(DataClassJsonMixin):
    protease: str = ("",)  # eg 'trypsin'
    labels: list[str] = ([],)  # eg ['K','Y','DE']

    def __post_init__(self):
        self.validate()

    def __repr__(self):
        return f"{self.protease}_{'_'.join(self.labels)}"

    def validate(self):
        assert self.protease in proteases, "Bad protease passed to ProteaseLabelScheme"
        assert all(
            [l.isalpha() and l.isupper() for l in self.labels]
        ), "Bad label passed to ProteaseLabelScheme"
        return gen_config.ValidationResult(True)


@dataclass
class SchemePermutator(DataClassJsonMixin):
    # We call a protease + some set of labels a "scheme"
    # This class is meant to allowing specification of
    # a set of proteases and labels that should be combined
    # to produce a list of schemes.  And/or a human may wish
    # to spec one or more specific schemes.

    def __post_init__(self):
        self.validate()

    proteases: Optional[list[str]] = gen_config.gen_field(
        help="A list of proteases to use for protease+labels schemes.",
        default_factory=list,
    )

    labels: list[str] = gen_config.gen_field(
        help="A list of AAs to potentially label, written as one or two-label combos, e.g. K,DE",
        default_factory=list,
    )

    n_channels: int = gen_config.gen_field(
        help="How many channels to generate schemes for.  E.g. if you specify 5 labels, and 3 channels, "
        'then you will end up with "5 choose 3" schemes (assuming only one protease was specified)',
        default=1,
    )

    user_schemes: list[ProteaseLabelScheme] = gen_config.gen_field(
        help="A user-specified list of ProteaseLabelScheme to use.",
        default_factory=list,
    )

    # This is not something a user is meant to supply, but it seems I have to follow the
    # same patter of gen_field() and provide help.  This doesn't really make sense to me.
    generated_schemes: list[ProteaseLabelScheme] = gen_config.gen_field(
        hidden=True,
        default_factory=list,
        help="holds schemes produced via gen_schemes()",
    )

    def gen_schemes(self):
        """
        From the given proteases, labels, and n_channels, generate the
        (labels choose n_channels) * proteases schemes.
        """
        self.generated_schemes = []
        label_combos = list(itertools.combinations(self.labels, self.n_channels))
        for p in self.proteases:
            for l in label_combos:
                self.generated_schemes.append(
                    ProteaseLabelScheme(protease=p, labels=list(l))
                )
        return self.generated_schemes

    def validate(self):
        assert all(
            [p in proteases for p in self.proteases]
        ), "Bad protease passed to SchemePermutator"
        assert all(
            [l.isalpha() and l.isupper() for l in self.labels]
        ), "Bad label passed to SchemePermutator"
        return gen_config.ValidationResult(True)


@dataclass
class SurveyV2Params(DataClassJsonMixin):
    # Maybe I'll inherit params from TestNN, but let's see what we want first...
    pass
