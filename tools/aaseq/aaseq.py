"""
Any Amino-acid sequence, might be a peptide or protein.
"""

import re
from io import StringIO
from typing import List

import pandas as pd

_legal_chars_pat = re.compile(r"^[A-Z0-9\.\[\]]$")

_split_pat = re.compile(
    r"""
    ([A-Z0-9\.])         # Start with a legal char
    (                    # Optionally followed by a [] expression: eg A[1]
        \[               # Open square bracket
        .*?              # Lazy characters inside the square bracket
        \]               # Close square bracket
    )?
    """,
    re.VERBOSE,
)

# 1 Letter code, 3 Letter code, Name, Molecular weight, Frequency
_iupac_codes_csv = """
    aa, triplet, name, weight, frequency
    A, Ala, Alanine, 71.04, 8.25
    B, Asx, Aspartic Acid or Asparagine, 0, 0
    C, Cys, Cysteine, 103.01, 1.37
    D, Asp, Aspartic Acid, 115.03, 5.45
    E, Glu, Glutamic Acid, 129.04, 6.75
    F, Phe, Phenylalanine, 147.07, 3.86
    G, Gly, Glycine, 57.02, 7.07
    H, His, Histidine, 137.06, 2.27
    I, Ile, Isoleucine, 113.08, 5.96
    J, Xle, Leucine or Isoleucine, 0, 0
    K, Lys, Lysine, 128.09, 5.84
    L, Leu, Leucine, 113.08, 9.66
    M, Met, Methionine, 131.04, 2.42
    N, Asn, Asparagine, 114.04, 4.06
    O, Pyl, Pyrrolysine, 0, 0
    P, Pro, Proline, 97.05, 4.70
    Q, Gln, Glutamine, 128.06, 3.93
    R, Arg, Arginine, 156.10, 5.53
    S, Ser, Serine, 87.03, 6.56
    T, Thr, Threonine, 101.05, 5.34
    U, Sec, Selenocysteine, 167.057, 0
    V, Val, Valine, 99.07, 6.87
    W, Trp, Tryptophan, 186.08, 1.08
    X, Xaa, Unspecified or unknown, 0, 0
    ., Xaa, Unspecified or unknown, 0, 0
    Y, Tyr, Tyrosine, 163.06, 2.92
    Z, Glx, Glutamic Acid or Glutamine, 0, 0
"""

aa_code_df = pd.read_csv(StringIO(_iupac_codes_csv), sep=",", skipinitialspace=True)
aa_code_df.frequency /= 100.0


def aa_str_to_list(seqstr, return_cleaned_str=False):
    """
    Convert from a string like "AB.D" to a list

    Arguments:
        seqstr: A string of the form:
            A-Z and "." are allowed
            Parens () are comments and are dropped including nested "A(hello())B" == ["A", "B"]
            Squares [] are modifiers and are bound to the previous amino-acid: "A[p]B" == ["A[p]", "B"]

    Returns:
        aa_list: List[strs]
        cleaned_str: The input without comments or whitespace.
    """

    cleaned_str = ""

    # REMOVE non-alphanum, parens and convert to uppercase
    paren_count = 0
    square_count = 0
    for i in list(seqstr):
        if i == "(":
            paren_count += 1
        elif i == ")":
            paren_count -= 1
        elif i == "[":
            square_count += 1
        elif i == "]":
            square_count -= 1

        if paren_count == 0:
            if square_count > 0:
                cleaned_str += i
            else:
                c = i.upper()
                if _legal_chars_pat.match(c):
                    cleaned_str += c
                elif c in (" ", "\t", "\r", "\n", "(", ")"):
                    # Strip whitespace.
                    pass
                else:
                    raise ValueError(f"Illegal character '{c}' found in AASeqParser.")

    if paren_count != 0:
        raise ValueError("Mismatching parentheses")

    if square_count != 0:
        raise ValueError("Mismatching square brackets")

    parts = re.findall(_split_pat, cleaned_str)
    seq_list = [part[0] + part[1] for part in parts]

    if return_cleaned_str:
        return seq_list, cleaned_str
    return seq_list


def aa_list_to_str(seqlist, spaces=None):
    if spaces is None:
        return "".join(seqlist)

    s = ""
    len_ = len(seqlist)
    for i in range(0, len_, spaces):
        s += "".join(seqlist[i : i + spaces]) + " "
    return s


def aa_random(n_aas: int) -> str:
    df = aa_code_df.sample(n_aas, replace=True, weights="frequency")
    return df.aa.str.cat()


def raap_list(n_peps: int, n_aas: int) -> List[str]:
    """
    Random Amino-Acids at Proportional Frequencies.
    Naming convention: "10 raap-5" means 10 randomly generated RAAP peptides of length 5
    """
    return [aa_random(n_aas) for _ in range(n_peps)]
