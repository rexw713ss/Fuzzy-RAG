"""Reading the DPR Wikipedia passages (spec v0.3, section 7.2).

psgs_w100.tsv(.gz) has a header row (id, text, title) and CSV-style quoting: a text that contains
quotes is wrapped in quotes with inner quotes doubled. It is parsed with csv.reader and a tab
delimiter, exactly as Contriever's own loader does, so the text we embed matches the text Meta
embedded.
"""

import csv
import gzip

CORPUS_SIZE = 21_015_324


def iter_passages(path, ids=None):
    """Yield (id, title, text) in file order.

    ids: optional set of passage ids (strings); only those are yielded, and reading stops as soon
    as all of them have been seen. Ids are unique in the corpus.
    """
    wanted = None if ids is None else set(ids)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        if header != ["id", "text", "title"]:
            raise ValueError(f"unexpected header {header}")
        for row in reader:
            if wanted is not None:
                if row[0] not in wanted:
                    continue
                wanted.discard(row[0])
            yield row[0], row[2], row[1]
            if wanted is not None and not wanted:
                return


def load_passages(path, ids):
    """{id: (title, text)} for the given ids. Raises if any id is missing from the corpus."""
    found = {pid: (title, text) for pid, title, text in iter_passages(path, ids)}
    missing = set(ids) - set(found)
    if missing:
        raise KeyError(f"{len(missing)} id(s) not in the corpus, e.g. {sorted(missing)[:3]}")
    return found
