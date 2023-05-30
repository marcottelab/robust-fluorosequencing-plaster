def fasta_split(fasta_str):
    """
    Splits a fasta file like:
        >sp|P10636|TFB_HUMAN Microtubule-associated protein tfb OS=Homo sapiens OX=9606 GN=MAPT PE=1 SV=5
        MAEPRQEFEVMEDHAGTYG
        SPRHLSNVSST
        >ANOTHER_TFB_HUMAN Microtubule-associated protein tfb OS=Homo sapiens OX=9606 GN=MAPT PE=1 SV=5
        ABC DEF
        GHI

    Returns:
        List(Tuple(header, sequence))
    """
    if fasta_str is None:
        fasta_str = ""

    lines = fasta_str.split("\n")

    groups = []
    group = []
    last_header = None

    def close(last_header):
        nonlocal group, groups
        group = [g.strip() for g in group if g.strip() != ""]
        if last_header is None and len(group) > 0:
            raise ValueError("fasta had data before the header")
        if last_header is not None:
            groups += [(last_header[1:], "".join(group))]
        group = []

    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            close(last_header)
            last_header = line
        else:
            group += [line]
    close(last_header)

    return groups
