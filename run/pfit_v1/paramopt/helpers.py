import plaster.genv2.generators.sim as simgen


def gen_protein_list(peptides, n_clones=0):
    def gen_clones(peptides, n_clones):
        return ["." * (k + 1) + x for k in range(0, n_clones) for x in peptides]

    peptides += gen_clones(peptides, n_clones)

    protein_list = [simgen.Protein(f"PEP{i}", seq) for i, seq in enumerate(peptides)]

    return protein_list
