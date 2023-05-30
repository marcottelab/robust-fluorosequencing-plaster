import gzip
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from plaster.root import cached_dir
from plaster.tools.aaseq.aaseq import aa_random


class RandomPeptides:
    """Class for generating RandomPeptides"""

    def __init__(self):
        self.proteins: list[str] = []
        self.fn: Optional[Path] = None

    def load(self, fn: Optional[Path] = None) -> None:
        """Load a cached Uniprot file of proteins to sample peptides from. Call this once before calling sample()."""

        if fn is None:
            fn = cached_dir() / "uniprot-human-proteome.fasta.gz"
            if not fn.exists():
                logging.warning(
                    f'{fn} does not exist. Try running "make fetch-human-proteome"'
                )
                return

        self.fn = fn

        self.proteins = []
        with gzip.open(fn, "rb") as f:
            header, seq = [""] * 2

            for ln in f:
                ln = ln.rstrip().decode("utf-8")
                if ln.startswith(">"):
                    if header and seq:
                        self.proteins += [(header, seq)]
                    header = ln
                    seq = ""
                else:
                    seq += ln

            if header and seq:
                self.proteins += [(header, seq)]

    def sample_uniprot(self, n: int, length: int) -> list[str]:
        """Sample n peptides of length length from loaded Uniprot proteins."""

        rv = []

        proteins = [x for x in self.proteins if len(x[1]) >= length]
        logging.debug(f"Sampling from {len(proteins)} of {len(self.proteins)}")

        lens = [len(x[1]) - length for x in proteins]
        lens_cs = np.cumsum(lens)

        for i in range(n):
            s = np.random.randint(0, lens_cs[-1] - length)

            ind = np.searchsorted(lens_cs, s, side="left", sorter=None)

            ind2 = s - lens_cs[ind - 1]

            p = proteins[ind][1][ind2 : ind2 + length]
            rv += [p]

        assert all([len(x) == length for x in rv])

        return rv

    def sample_random(self, n: int, length: int) -> list[str]:
        """Generate random n peptides of length length."""

        rv = [aa_random(length) for _ in range(n)]
        return rv

    def sample(self, n: int, length: int) -> list[str]:
        """Return n peptides of length length, either from a random source
        or from a load()ed set of proteins."""

        if self.proteins:
            return self.sample_uniprot(n, length)
        else:
            return self.sample_random(n, length)
