"""Tests for reading the DPR passage file (spec v0.3, section 7.2)."""

import gzip

import pytest

from main_logic import corpus as cp

# The first rows of psgs_w100.tsv, in the file's own quoting, plus a plain row.
SAMPLE = (
    'id\ttext\ttitle\n'
    '1\t"Aaron Aaron ( or ; ""Ahärôn"") is a prophet"\tAaron\n'
    '2\tGod at Sinai granted Aaron the priesthood\tAaron\n'
    '3\tA third passage\t"Title, with ""quotes"""\n'
)


@pytest.fixture(params=["plain", "gz"])
def corpus_file(tmp_path, request):
    if request.param == "gz":
        path = tmp_path / "psgs.tsv.gz"
        with gzip.open(path, "wt", encoding="utf8", newline="") as f:
            f.write(SAMPLE)
    else:
        path = tmp_path / "psgs.tsv"
        path.write_text(SAMPLE, encoding="utf8", newline="")
    return path


def test_quotes_are_unescaped_like_csv(corpus_file):
    rows = list(cp.iter_passages(corpus_file))
    assert rows[0] == ("1", "Aaron", 'Aaron Aaron ( or ; "Ahärôn") is a prophet')
    assert rows[2] == ("3", 'Title, with "quotes"', "A third passage")


def test_selected_ids_only(corpus_file):
    assert [r[0] for r in cp.iter_passages(corpus_file, ids={"3", "1"})] == ["1", "3"]


def test_load_passages_and_missing_ids(corpus_file):
    got = cp.load_passages(corpus_file, {"2"})
    assert got == {"2": ("Aaron", "God at Sinai granted Aaron the priesthood")}
    with pytest.raises(KeyError, match="not in the corpus"):
        cp.load_passages(corpus_file, {"2", "99"})


def test_wrong_header_rejected(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("id\ttitle\ttext\n1\ta\tb\n", encoding="utf8")
    with pytest.raises(ValueError, match="unexpected header"):
        list(cp.iter_passages(path))
