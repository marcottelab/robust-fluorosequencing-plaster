import typing

import pandas as pd

import plaster.genv2.generators.sim as simgen


def protein_df_to_list(df: pd.DataFrame) -> typing.List[simgen.Protein]:
    protein_list = []

    for idx in df.index:
        protein_list.append(
            simgen.Protein(
                name=df.loc[idx, "Name"],
                sequence=df.loc[idx, "Seq"],
                abundance=df.loc[idx, "Abundance"],
                is_poi=bool(df.loc[idx, "POI"]),
                ptm_locs="",
            )
        )

    return protein_list
