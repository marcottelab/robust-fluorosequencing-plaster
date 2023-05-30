import io
import logging
import urllib.parse

import pandas as pd
import requests as r

# UniProt REST API documentation
# https://www.uniprot.org/help/api
# https://www.uniprot.org/help/api_queries
# https://rest.uniprot.org/docs/?urls.primaryName=idmapping

_log = logging.getLogger(__name__)


def query_url(seq_accs_en: list[str]) -> str:
    return f"https://rest.uniprot.org/uniprotkb/accessions?accessions={','.join(seq_accs_en)}&fields=accession,id,sequence&format=tsv"


def get_seqs(seq_accs: list[str], batch_size=1000) -> pd.DataFrame:
    if 0 == len(seq_accs) or 0 == batch_size:
        return pd.DataFrame(columns=["accession", "id", "sequence"])

    dfs = [
        get_seqs_batch(seq_accs[idx : idx + batch_size])
        for idx in range(0, len(seq_accs), batch_size)
    ]

    return pd.concat(dfs, axis=0)


def get_seqs_batch(seq_accs: list[str]) -> pd.DataFrame:
    """
    Return a list of sequences given a list of UniProt accession IDs.

    Parameters
    ----------
        seq_accs: A list of UniProt accessions.

    Returns
    -------
        sequences: A dataframe containing rows of accession, id, and sequence.

    """

    s = r.Session()
    seq_accs_en = [urllib.parse.quote(x) for x in seq_accs]
    url = query_url(seq_accs_en)

    _log.debug(f"get_seqs.query: {url}")

    response = s.post(url, allow_redirects=True)

    if not response.ok:
        # strip the bad seqs and try again

        try:
            # Assuming error response looks like:
            # Accession 'FAILED' has invalid format. It should be a valid UniProtKB accession.
            bad_seq_accs = set([x.split("'")[1] for x in response.text.split("\n")[1:]])
        except IndexError:
            _log.critical(f"failed to parse {response.text}")
        else:
            # If the responses are not URL encoded, good_seq_accs may not be correct.
            good_seq_accs = [x for x in seq_accs_en if x not in bad_seq_accs]
            retry_url = query_url(good_seq_accs)

            _log.warning(
                f"initial query failed with bad accessions: {','.join(bad_seq_accs)}"
            )
            _log.debug(f"retrying with query: {retry_url}")

            response = s.post(retry_url, allow_redirects=True)

    if not response.ok:
        _log.warning("final query failed")
        return pd.DataFrame(columns=["accession", "id", "sequence"])

    return pd.read_csv(
        io.StringIO(response.text),
        sep="\t",
        header=0,
        names=["accession", "id", "sequence"],
    )
