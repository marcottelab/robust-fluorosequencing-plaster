from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import marshmallow
import pandas as pd
from dataclasses_json import DataClassJsonMixin, config

from plaster.run.base_result_dc import (
    BaseResultDC,
    LazyField,
    generate_flyte_result_class,
    lazy_field,
)
from plaster.run.survey_v2.survey_v2_params import ProteaseLabelScheme, SurveyV2Params


class MultiPeptideMetric(Enum):
    DIST_AVG = "dist_avg"
    DIST_MIN = "dist_min"


class Objective(Enum):
    PROTEIN_ID = "protein_id"
    COVERAGE = "coverage"
    PTM = "ptm"


@dataclass
class SurveyV2ResultDC(BaseResultDC, DataClassJsonMixin):
    params: SurveyV2Params
    schemes: list[ProteaseLabelScheme]
    _survey: LazyField = lazy_field(pd.DataFrame)

    survey_columns = [
        "pro_i",
        "pep_i",
        "pep_start",
        "pep_stop",
        "pep_len",
        "seqstr",
        "P2",
        "flustr",
        "n_dyes_max_any_ch",
        "flu_count",
        "nn_pep_i",
        "nn_dist",
    ]

    allow_proline_at_2: bool = False
    run_exclude: List[str] = field(default_factory=list)
    run_include: List[str] = field(default_factory=list)
    max_dyes_per_ch: int = -1
    max_pep_len: int = -1
    max_ptms_per_pep: int = -1
    multi_peptide_metric: Optional[MultiPeptideMetric] = field(
        default=None,
        metadata=config(
            mm_field=marshmallow.fields.String(),
            encoder=lambda x: x.value if x is not None else "",
            decoder=lambda s: MultiPeptideMetric(s) if s != "" else None,
        ),
    )
    multi_pro_rank: str = MultiPeptideMetric.DIST_MIN.value
    n_best_schemes: int = 50
    n_peps_per_scheme: int = 1
    objective: Objective = field(
        default=Objective.PROTEIN_ID,
        metadata=config(
            mm_field=marshmallow.fields.String(),
            encoder=lambda x: x.value,
            decoder=Objective,
        ),
    )
    poi_only: bool = False
    pro_subset: List[str] = field(default_factory=list)
    ptm_subset: List[int] = field(default_factory=list)

    @property
    def survey(self):
        return self._survey.get()

    def _domain_loss(self, df, filters, msg):  # debug aid
        if filters.verbose:
            if filters.objective == Objective.PTM:
                domain_loss = set(filters.requested_ptms) - set(
                    list(df.ptm.astype(int))
                )
                # print(filters.requested_ptms)
            else:
                domain_loss = set(filters.requested_proteins) - set(list(df.pro_i))
                # print(filters.requested_proteins)

            if domain_loss:
                print(
                    f"  {filters.objective.value} {msg} domain_loss: {sorted(domain_loss)}"
                )

    def _apply_filters(self, filters, prep=None):
        """
        filters may be used to reduce the entries rows of _survey.
        # TODO: can we just require prep to avoid all the exception logic below?
        """
        if filters.verbose:
            print(f"\n{(self._folder / '..' / '..').name}")

        df = self.survey

        # If the caller is optimizing for PTMs, we need to add PTM information for
        # the peptides.  This is done here so that protein PTMs can be changed
        # after a run is complete.  the PrepResult is required.  Note the inner
        # join which causes the resulting df to only contain entries which have
        # PTM locations specified.
        if filters.objective == Objective.PTM:
            if prep:
                peps__ptms = prep.peps__ptms(ptm_peps_only=True, ptms_to_rows=True)[
                    ["pep_i", "n_pep_ptms", "ptm"]
                ]
                if len(peps__ptms) > 0:
                    df = (
                        df.set_index("pep_i")
                        .join(peps__ptms.set_index("pep_i"), how="inner")
                        .reset_index()
                    )
                # Write down which ptms were explicitly or implicitly requested by the caller,
                # so we can know later which ones were removed by filtering.  Note that looking
                # at the unique values in df.ptm is not quite right if more than one protein has a
                # PTM at the same location.  In theory this is OK, but it means our PTM accounting
                # for "domain_loss" for PTMs is not quite right, so assert here and deal with that
                # if necessary.
                assert len(df.ptm) == len(
                    df.ptm.unique()
                ), "More than one protein has the same PTM location?"
                filters.requested_ptms = filters.ptm_subset or sorted(
                    list(df.ptm.unique().astype(int))
                )
                if filters.verbose:
                    print(f"  ptms domain: {filters.requested_ptms}")
            else:
                raise ValueError("Must supply PrepResult to optimize for PTMs")

        if filters is not None:
            # Do protein subset or POI which substantially reduces df
            if len(filters.pro_subset) > 0:
                if prep is None:
                    raise ValueError("Must supply PrepResult to filter by pro_subset")
                pros = prep.pros()
                pro_iz = pros[pros.pro_id.isin(filters.pro_subset)].pro_i.values
                df = df[df.pro_i.isin(pro_iz)]
            if filters.poi_only == True:
                if prep is None:
                    raise ValueError(
                        "Must supply PrepResult to filter by proteins-of-interest"
                    )
                poi_iz = prep.pros__pois().pro_i.values
                if len(poi_iz) > 0:
                    # If there are no entries, then all are considered "of interest",
                    # so only filter here if there are some specifically marked.
                    df = df[df.pro_i.isin(poi_iz)]

            # Write down requested proteins so we can tell the user which ones got
            # removed by filtering.
            filters.requested_proteins = sorted(list(df.pro_i.unique()))
            if filters.verbose:
                print(f"  proteins domain: {filters.requested_proteins}")

            self._domain_loss(df, filters, "post-protein-filtering")

            # remove rows per filtering
            if filters.max_pep_len is not None:
                df = df[df.pep_len <= filters.max_pep_len]
                self._domain_loss(df, filters, "max_pep_len")
            if filters.max_dyes_per_ch is not None:
                df = df[df.n_dyes_max_any_ch <= filters.max_dyes_per_ch]
                self._domain_loss(df, filters, "max_dyes_per_ch")
            if filters.max_ptms_per_pep is not None:
                df = df[df.n_pep_ptms <= filters.max_ptms_per_pep]
                self._domain_loss(df, filters, "max_ptms_per_pep")
            if len(filters.ptm_subset) > 0:
                # WARNING: this affects PTMs for ALL proteins that have them.
                # This is typically OK, since you're looking for PTMs on a single
                # protein of interest, but if you had a PTM at location 100 on two
                # different proteins, this filter would apply to both of them.
                df = df[df.ptm.astype(int).isin(filters.ptm_subset)]
                self._domain_loss(df, filters, "ptm_subset")
            if not filters.allow_proline_at_2:
                df = df[df.P2 == False]
                self._domain_loss(df, filters, "allow_proline_at_2")

        return df.copy()

    def n_uniques(self, filters=None, df=None):
        """
        Returns number of peptides with unique flus.  This is probably actually not
        very interesting if you are comparing different proteases, but it may be if
        you are comparing different labeling schemes for a single protease, or for
        pre-specified peptide sets like MHC.
        """
        df = self._apply_filters(filters) if df is None else df
        return len(df[df.flu_count == 1])

    def protein_coverage(self, prep_result, filters=None, df=None):
        """
        Returns the percentage coverage of proteins with peptides that have unique flus.
        If any proteins are marked "of interest" via the pro_report flag, then we
        compute the coverage only for those proteins, else all proteins are used.
        """
        df = self._apply_filters(filters, prep=prep_result) if df is None else df
        df = df[df.flu_count == 1]  # only use peptides that have unique flus
        n_poi = prep_result.n_pois
        poi_iz = (
            prep_result.pros__pois().pro_i.values
            if n_poi > 0
            else prep_result.pros().pro_i.values
        )

        # OLD - returns average coverage of proteins in domain
        # poi_percent_coverage = np.zeros_like(poi_iz).astype(float)
        # proseq_groups = prep_result.proseqs().groupby("pro_i")
        # pep_coverage_groups = df.groupby("pro_i")
        # for i, poi_i in enumerate(poi_iz):
        #     try:
        #         poi_percent_coverage[i] = (
        #             pep_coverage_groups.get_group(poi_i).pep_len.sum()
        #             / proseq_groups.get_group(poi_i).aa.count()
        #         )
        #     except KeyError:
        #         pass  # protein not covered at all by peps
        # avg_coverage = np.mean(poi_percent_coverage)
        # return avg_coverage

        # NEW - returns total percentage coverage of multiple proteins
        # in the case multiple proteins in domain of interest.
        proseq_groups = prep_result.proseqs().groupby("pro_i")
        pep_coverage_groups = df.groupby("pro_i")
        total_aa_covered = 0
        total_proteins_len = 0
        for poi_i in poi_iz:
            try:
                total_aa_covered += pep_coverage_groups.get_group(poi_i).pep_len.sum()
            except KeyError:
                pass  # protein not covered at all, no length added to total_aa_covered
            total_proteins_len += proseq_groups.get_group(poi_i).aa.count()
        return total_aa_covered / total_proteins_len

    def max_nn_dist(self, unique_flus_only=True, filters=None, df=None):
        """
        Returns the maximum nearest-neighbor distance over all perfect dyetracks
        from the set of peptides.  We will probably want more nuanced information
        here, or via some other fn -- something that gets at more than just the
        max, perhaps including information for the top N, and some measure of
        'separated-ness' across that set.  A single non-normalized max value feels
        kind of fragile.
        """
        df = self._apply_filters(filters) if df is None else df
        if unique_flus_only:
            df = df[df.flu_count == 1]
        return df.nn_dist.max()

    def max_nn_dist_peps(self, prep=None, unique_flus_only=True, filters=None, df=None):
        """
        Like max_nn_dist(), but instead of returning just the max dist, returns information
        about the peptide(s) as well, in a DataFrame.  filters.n_peps_per_scheme controls how
        many rows are returned for non-ptm filtering.  For ptm-filtering, the number of peps
        is determined by how many peptides in this scheme contain ptms -- each will be
        returned.

        prep : a PrepResult.  If provided, we'll include the protein_coverage in the results.
        """
        df = self._apply_filters(filters, prep=prep) if df is None else df
        if unique_flus_only:
            df = df[df.flu_count == 1]
            self._domain_loss(df, filters, "unique_flus_only")

        if prep is not None:
            cols = list(df.columns)
            cols.insert(1, "pro_id")
            df = (
                df.set_index("pro_i")
                .join(prep.pros().set_index("pro_i"), how="left")
                .reset_index()[cols]
            )
            df["nn_coverage"] = self.protein_coverage(prep, df=df)

        df["nn_unique"] = self.n_uniques(df=df)

        df = df.sort_values(
            by=["nn_dist", "pep_len", "n_dyes_max_any_ch"],
            ascending=[False, True, True],
        )

        if filters.objective != Objective.PTM:
            # If we only need to know a single max dist across all peps/proteins, we're done.
            # This is the case if multi_peptide_metric is None - we're not trying to take into
            # account the performance on multiple protein distances.  Return the n best peps.
            if filters.multi_peptide_metric is None:
                return df[: filters.n_peps_per_scheme].reset_index(drop=True)

            # Otherwise we need a composite metric that considers the nn_dist
            # of peptides from multiple proteins.  To start, we rank
            # the peptides from each protein based on the sort order already
            # established above, and take the top n based on filters.n_peps_per_scheme,
            # leaving us with the top n peptides, ranked, from each protein
            df["nn_rank"] = (
                df.groupby("pro_i").nn_dist.rank("first", ascending=False).astype("int")
            )
            df = df[df.nn_rank.isin(range(filters.n_peps_per_scheme + 1))]

            # Then we compute a couple of composite metrics which are fns
            # of the nn_dist from the peptides of each rank.  The caller
            # can sort on these across multiple runs.
            df["nn_dist_avg"] = df.groupby("nn_rank").nn_dist.transform("mean")
            df["nn_dist_min"] = df.groupby("nn_rank").nn_dist.transform("min")

        else:
            # For PTMs, we have already filtered out the peptides that
            # don't contain PTMs, so we just need these composite metrics
            # computed for the single set of all of the peptides in the df.
            df["nn_rank"] = 1
            df["nn_dist_avg"] = df.nn_dist.mean()
            df["nn_dist_min"] = df.nn_dist.min()

        # It can be that the caller is interested in N proteins or PTMs, but some
        # of those have been lost due to filtering etc.  This will cause the
        # mean and stats above to be w.r.t. too few set members, so adjust
        # these in a way that makes sense.  In the initial application of
        # filtering, the number of proteins/ptms the caller is interested in
        # has been saved. (If a protein or PTM was "lost" due to filtering,
        # it effectively merged with the background, is not observable,
        # and its nn_dist is therefore 0 -- indistinguishable from some
        # neighbor)
        filter_pass = 1.0
        domain_loss = ""  # either lost proteins, or lost ptms
        if filters.objective == Objective.PTM:
            filter_pass = len(df) / len(filters.requested_ptms)
            domain_loss = set(filters.requested_ptms) - set(list(df.ptm.astype(int)))
            if filters.verbose and domain_loss:
                print(f"  ** final domain_loss: {sorted(domain_loss)}")
        elif len(filters.requested_proteins) == 0:
            filter_pass = 0
        else:
            filter_pass = len(df.pro_i.unique()) / len(filters.requested_proteins)
            domain_loss = set(filters.requested_proteins) - set(list(df.pro_i))

        domain_loss = str(sorted(domain_loss)) if domain_loss else ""
        df["domain_loss"] = domain_loss
        assert filter_pass <= 1.0
        if filter_pass != 1.0:
            df.nn_dist_avg *= 1.0 - filter_pass
            df.nn_dist_min = 0

        if filters.verbose:
            print(f"  filter_pass is {filter_pass}")

        return df.sort_values(
            by=["nn_rank", "pro_i"], ascending=[True, True]
        ).reset_index(drop=True)

    def nn_stats(self, prep_result, filters=None):
        """
        Returns a tuple that gives the main stats for this survey run
        that can be used to pick from a list of such survey runs:
        nn_uniques - the number of unique peptides
        nn_coverage - percent coverage of protein(s) by unique peptides
        nn_dist - distance to neighbor for most isolated dyetrack
        """
        df = self._apply_filters(filters, prep=prep_result)
        n_uniques = (self.n_uniques(df=df),)
        return (
            self.n_uniques(df=df),
            self.protein_coverage(prep_result, df=df),
            self.max_nn_dist(df=df),
        )


SurveyV2FlyteResult = generate_flyte_result_class(SurveyV2ResultDC)
