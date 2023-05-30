# This should become a utility function in library code: build_protein_list(csv_filename)
# and will eventually get called from the Generator for Simulations, so that a user can
# just specify a CSV file and not worry about this boilerplate work.

from pathlib import Path

import pandas as pd
import structlog

import plaster.genv2.generators.sim as simgen
import plaster.tools.aaseq.uniprot as uniprot

logger = structlog.getLogger()


def _build_peptide_list(
    csv_seqs: Path, csv_names: Path, poi_names: list[str] = []
) -> tuple[list[simgen.Protein], pd.DataFrame]:
    """
    Builds the protein_list as needed for SimConfig, given a CSV with sequences
    and CSV file with names, as was the case for the marvel5 data.
    """

    seqs_df = pd.read_csv(csv_seqs, names=["sequence", "pep_i"])
    names_df = pd.read_csv(csv_names, names=["Name", "pep_i"])
    proteins_df = pd.merge(names_df, seqs_df, on="pep_i")
    proteins_df["POI"] = proteins_df["Name"].isin(poi_names).astype(bool)

    protein_list = []
    for i, row in proteins_df.iterrows():
        protein_list.append(
            simgen.Protein(
                name=row["Name"],
                sequence=row["sequence"],
                abundance=1,
                is_poi=row["POI"],
                ptm_locs=None,
            )
        )
    return protein_list, proteins_df


def _build_protein_list(
    csv: Path,
    poi_names: list[str] = [],
    use_abundance: bool = False,
    use_ptms: bool = True,
) -> tuple[list[simgen.Protein], pd.DataFrame]:
    """
    Builds the protein_list as needed for SimConfig, given a CSV file
    with the columns:
        Required: UniprotAC or Sequence
        Optional: Abundance, POI, and PTM
    Checks for required columns and drops unknown columns.
    """

    known_columns = {"Name", "UniprotAC", "Sequence", "Abundance", "POI", "PTM"}
    required_columns = set([])
    proteins_df = pd.read_csv(csv, dtype={"POI": bool})
    columns = set(proteins_df.columns)

    missing = required_columns - columns
    unknown = columns - known_columns
    if missing:
        msg = f"missing required columns in protein CSV {csv.name}: {missing}"
        logger.error(msg)
        raise ValueError(msg)
    if unknown:
        logger.warning(f"dropping unknown columns in protein CSV {csv.name}: {unknown}")
        proteins_df.drop(labels=list(unknown), axis=1, inplace=True)

    if "UniprotAC" in columns:
        # Get sequences from Uniprot
        req_seq = proteins_df.UniprotAC.values
        seq_df = uniprot.get_seqs(req_seq)
        seq_df.rename(columns={"accession": "UniprotAC"}, inplace=True)
        seq_df.drop(labels=["id"], axis=1, inplace=True)

        # Merge Uniprot and provided sequences into our df
        proteins_df = proteins_df.merge(seq_df, on="UniprotAC")

        # Match up UniprotAC and Sequence formats
        proteins_df.drop(labels=["Name"], axis=1, inplace=True)
        proteins_df.rename(
            columns={"UniprotAC": "Name", "sequence": "Sequence"}, inplace=True
        )
    elif "Sequence" in columns:
        if "Name" not in columns:
            proteins_df["Name"] = [f"Protein_{x:04d}" for x in range(len(proteins_df))]
    else:
        msg = f"Neither UniprotAC nor Sequence columns found in protein CSV {csv.name}"
        logger.error(msg)
        raise ValueError(msg)

    # Move any requested proteins to top for convenience, and mark these
    # as POI, setting others to false (overriding the order and POI setting
    # of the CSV file.)
    if poi_names:
        count = len(poi_names)
        # create a new col to allow us to sort this many to the top
        proteins_df["new_idx"] = range(count, len(proteins_df) + count)
        # reset all POI values to False
        proteins_df["POI"] = False
        for i, pro_name in enumerate(poi_names):
            try:
                proteins_df.loc[proteins_df["Name"] == pro_name, ["new_idx", "POI"]] = [
                    i,
                    True,
                ]
            except:
                logger.error(f"Failed re-index of proteins_df", pro_name=pro_name)
                raise KeyError
        proteins_df = (
            proteins_df.sort_values("new_idx")
            .reset_index(drop="True")
            .drop("new_idx", axis=1)
        )

    # Rebuild columns variable in case the names have changed
    columns = set(proteins_df.columns)
    has_abundance = "Abundance" in columns
    has_ptms = "PTM" in columns
    has_poi = "POI" in columns
    use_abundance = use_abundance and has_abundance
    use_ptms = use_ptms and has_ptms

    # Make a list of Protein objects as required by SimConfig, optionally
    # using the abundance information from the CSV.
    protein_list = []
    for i, row in proteins_df.iterrows():
        protein_list.append(
            simgen.Protein(
                name=row["Name"],
                sequence=row["Sequence"],
                abundance=row["Abundance"] if use_abundance else 1,
                is_poi=row["POI"] if has_poi else True,
                ptm_locs=row["ptm_locs"] if use_ptms else None,
            )
        )

    return protein_list, proteins_df
