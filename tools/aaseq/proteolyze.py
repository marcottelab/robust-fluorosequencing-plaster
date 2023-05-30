proteases = dict(
    # Protease rules based on:
    # https://web.expasy.org/peptide_cutter/peptidecutter_enzymes.html#exceptions
    #
    # Examples:
    #   [(True, "K;")] => Cuts c-side of K
    #   [(True, ";D")] => Cuts n-side of D
    #
    #   Note, the last matching rule wins:
    #   [
    #       (True, "     KR;   "),
    #       (False, "    KR; P "),
    #   ] => Cuts c-side of K or R unless followed by P (because the second rule wins)
    trypsin=[
        (True, "     KR;   "),
        (False, "    KR; P "),
        (True, " W , K ; P "),
        (True, " M , R ; P "),
        (False, "CD, K ; D "),
        (False, "C,  K ; HY"),
        (False, "C , R ; K "),
        (False, "R , R ; HR"),
    ],
    lysc=[(True, "K;")],
    lysn=[(True, ";K")],
    argc=[(True, "R;")],
    aspn=[(True, ";D")],
    gluc_ph4=[(True, "E;"), (False, "E;PE")],
    gluc_ph8=[(True, "DE;"), (False, "DE;PE")],
    # Less tools...
    cnbr=[(True, "M;")],
    idosobenzoic=[(True, "W;")],
    ntcb=[(True, ";C")],
    endopro=[(True, "AP;")],
    lysarg=[(True, ";K"), (True, ";R")],
)


def _normalize_protease_rules(rule):
    """
    Given a rule like: "AB, C; D, EF, G" convert to:
        ["", "", "AB", C",    "D", "EF", "G", ""]
    """
    n_side, c_side = rule.split(";")
    n_side = [i.strip() for i in n_side.split(",")]
    c_side = [i.strip() for i in c_side.split(",")]
    eight_tuple = [""] * (4 - len(n_side)) + n_side + c_side + [""] * (4 - len(c_side))
    assert len(eight_tuple) == 8
    return eight_tuple


def compile_protease_rules(protease):
    """
    Compiled rules are a list of rules in the form:
        (True, ["", "", "AB", C",    "D", "EF", "G", ""]),

    Where, in this example, the leading True is an example of a "keep=True" rule followed by an 8-tuple.
    """
    compiled = [
        (keep, _normalize_protease_rules(rule)) for keep, rule in proteases[protease]
    ]
    return compiled


def cleavage_indices_from_rules(pro_seq_df, rules):
    """
    Test each rule in order for each position of the pseq
    and return the indices of the cleave sites.

    Returns:
        A list of indices of where the protease will cut (cut is BEFORE this position).

    Example:
        Suppose we cut after B
        pro_seq_df:
            0   A
            1   B
            2   C
            3   B
            4   D
            5   E
        returns:
            [2, 4]

    TASK:
        If this is slow, converting it to dataframe operation might make sense
    """
    cleave_i = []

    padded = ["."] * 4 + pro_seq_df.aa.values.tolist() + ["."] * 4

    for start_i in range(1, len(pro_seq_df)):
        cleave = False
        for keep, rule in rules:
            assert len(rule) == 8

            # At the current start position evaluate the rule at all eight rule positions
            # If all 8 positions match then this is a match.
            # If it is a match the "keep" flag says if this is a keep or reject
            match = [
                rule[i] == "" or padded[start_i + i][0] in rule[i] for i in range(8)
            ]
            if all(match):
                cleave = keep

        if cleave:
            # If the last matching rule said to keep this cleavage then we add it to the list
            cleave_i += [start_i]

    return cleave_i
