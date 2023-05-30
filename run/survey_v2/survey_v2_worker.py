from plaster.run.survey_v2.c import survey_v2 as survey_v2_fast
from plaster.run.survey_v2.survey_v2_result import SurveyV2ResultDC
from plaster.tools.aaseq.aaseq import aa_str_to_list
from plaster.tools.pipeline.pipeline import Progress
from plaster.tools.zap import zap

'''
def dist_to_closest_neighbors(dyemat, dytpeps):
    """
    Compute euclidean distance to the closest neighbor of
    all dye-tracks produced by a simulation.

    Returns:
        DataFrame( "pep_i", "collision_metric" )
        A low value of collision_metric means that the peptide is
        well isolated from other peptides (ie good)
    """

    pyflann.set_distance_type("euclidean")
    flann = pyflann.FLANN()
    flann.build_index(dyemat, algorithm="kdtree_simple")

    dye_nn_iz, dye_nn_dists = flann.nn_index(dyemat, num_neighbors=2)

    dytpep_df = pd.DataFrame(dytpeps, columns=["dye_i", "pep_i", "n_reads"])
    dye_nn_df = pd.DataFrame(
        dict(
            dye_i=np.arange(dye_nn_iz.shape[0]),
            dye_nn_i=dye_nn_iz[:, 1],  # 1 because col 0 is always the point itself
            dist=np.sqrt(dye_nn_dists[:, 1]),
        )
    )

    df = (
        dytpep_df.set_index("dye_i")
        .join(dye_nn_df.set_index("dye_i"), how="left")
        .reset_index()
        .sort_values(by="pep_i")
    )

    group = df.groupby("pep_i")

    sums = group.n_reads.transform(np.sum)
    df["collision_metric"] = df.n_reads * sums / df.dist

    # df is now combining the pep_i, dye_i, n_reads with the nn_i and a distance.
    # We seek to judge each peptide for "how well spaced it is from others".
    # For each of the dyetracks that compose a peptide we scale the distances
    # by the fraction of each dyetrack is contributing to that peptide
    #
    # Example:
    #   pep1 comes from dye3 10 times and dye5 20 times.
    #   dye3's closest neighbor is 2 units away
    #   dye5's closest neighbor is 1 unit away
    #   10 / 2 + 20 / 1 = 5 + 20 = 25
    # Compare to:
    #   pep2 comes from dye5 10 times and dye7 20 times.
    #   dye5's closest neighbor is (still) 1 unit away
    #   dye7's closest neighbor is 1 units away
    #   10 / 1 + 20 / 1 = 10 + 20 = 30
    #
    # Note that we're adding all of the terms for each contributing
    # dyetrack and the more we add the more we "muddled" we consider
    # this dyetrack. By adding the inverses the lowest number
    # becomes the most well separated peptide.

    # We really want to know the distance to the closest dyetrack that is NOT this peptide

    pep_df = pd.DataFrame(group.collision_metric.sum()).reset_index()

    np.save("dyemat.npy", dyemat)
    dytpep_df.to_pickle("dytpep_df.pkl")
    dye_nn_df.to_pickle("dye_nn_df.pkl")
    df.to_pickle("df.pkl")
    pep_df.to_pickle("pep_df.pkl")

    return pep_df
'''


def survey_v2(
    survey_v2_params, prep_result, sim_v3_result, result_class=SurveyV2ResultDC
):
    """
    Compute a distance between between peptides that exist in prep_result
    using the dye-tracks employed by nearest-neighbor.  Create a DF that
    collects these distances with other information useful in surveying
    a number of protease-label schemes to determine which ones are well
    suited to some informatics objective, such as identifying a protein(s).

    Notes:
        - We are including decoys if present.
    """

    with Progress("survey") as progress:

        pep_i_to_mic_pep_i, pep_i_to_isolation_metric = survey_v2_fast.survey(
            prep_result.n_peps,
            sim_v3_result.train_dytmat,
            sim_v3_result.train_dytpeps,
            n_threads=zap.get_cpu_limit(),
            progress=progress,
        )

        # Join this to some flu information so we have it all in one place, especially
        # info about degeneracy (when more than one pep has the same dyetrack)
        # This isn't very DRY, since this data already lives in the prep and sim results.
        # But it makes downstream report-code simpler and faster to filter and search
        # these results if everything you need is already joined in one DF.
        # My approach is to put everything into the SurveyResult that you want
        # to be able to filter on to minimize computation in the report.
        # This is possible for nearly everything, except things you want to
        # be able to change at report time, like what PTMs you're interested in
        # if this survey involves PTMs.
        #

        peps__flus = sim_v3_result.peps__flus(prep_result)
        peps__flus["pep_len"] = peps__flus.apply(
            lambda x: x.pep_stop - x.pep_start - 1, axis=1
        )

        peps__flus["nn_pep_i"] = pep_i_to_mic_pep_i
        peps__flus["nn_dist"] = pep_i_to_isolation_metric

        # include the peptide sequence, and whether it has Proline at position 2
        pepstrs = prep_result.pepstrs()
        pepstrs["P2"] = pepstrs.apply(
            lambda row: True
            if row.seqstr
            and len(row.seqstr) > 1
            and aa_str_to_list(row.seqstr)[1] == "P"
            else False,
            axis=1,
        )

        df = (
            peps__flus.set_index("pep_i")
            .join(pepstrs.set_index("pep_i"), how="left")
            .reset_index()
        )

        return result_class(params=survey_v2_params, schemes=[], _survey=df)


def short_survey_v2(survey_v2_params, prep_result, sim_v2_result):
    """Carry out the NN survey calc and leave the rest for postprocessing."""
    return survey_v2_fast.survey(
        prep_result.n_peps,
        sim_v2_result.train_dytmat,
        sim_v2_result.train_dytpeps,
        n_threads=zap.get_cpu_limit(),
        progress=None,
    )
