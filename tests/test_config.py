"""config.yaml must equal the constants the code uses (spec v0.3, 7.5).

If a value changes in one place and not the other, a test here fails.
"""

from pathlib import Path

import pytest

from main_logic import actions, bm25, calibration, dense, fuzzy, routing, signals, splits
from main_logic.config import load_config
from scripts import bm25_search, smoke_retrieval

CFG = load_config()


@pytest.mark.parametrize("section, key, constant", [
    ("retrieval", "dense_model", dense.MODEL),
    ("retrieval", "dense_revision", dense.REVISION),
    ("retrieval", "dense_max_length", dense.MAX_LENGTH),
    ("retrieval", "bm25_k1", bm25.K1),
    ("retrieval", "bm25_b", bm25.B),
    ("signals", "k", signals.K),
    ("signals", "pool", signals.POOL),
    ("signals", "rbo_p", signals.RBO_P),
    ("signals", "tau", signals.TAU),
    ("fuzzy", "centres", list(fuzzy.CENTRES)),
    ("fuzzy", "w_default", fuzzy.W_DEFAULT),
    ("fuzzy", "w_max", fuzzy.W_MAX),
    ("fuzzy", "n_out", fuzzy.N_OUT),
    ("routing", "theta1_range", list(routing.T1_RANGE)),
    ("routing", "theta2_range", list(routing.T2_RANGE)),
    ("routing", "step", routing.STEP),
    ("routing", "quality", routing.QUALITY),
    ("routing", "cost", routing.COST),
    ("actions", "a2_rerank", actions.A2_RERANK),
    ("calibration", "quantile", calibration.QUANTILE),
    ("calibration", "cluster_size", calibration.CLUSTER_SIZE),
    ("splits", "seed", splits.SEED),
    ("splits", "names", list(splits.SPLITS)),
])
def test_config_matches_code(section, key, constant):
    assert CFG[section][key] == constant


def test_evaluation_values_match_the_spec():
    """No code constant yet; step 0i adds them and moves these checks to them."""
    assert CFG["evaluation"]["oracle_delta"] == 0.02           # 6.1
    assert CFG["evaluation"]["bootstrap_resamples"] == 1000    # 6.3


def test_data_paths_match_the_scripts():
    data = CFG["data"]
    assert Path(data["root"]) == smoke_retrieval.DATA
    assert Path(data["bm25_index"]) == bm25_search.INDEX
    assert Path(data["corpus"]) == smoke_retrieval.DATA / "dpr" / "psgs_w100.tsv.gz"
    assert Path(data["embeddings"]) == (smoke_retrieval.DATA / "contriever-msmarco"
                                        / "wikipedia_embeddings")


def test_fitted_values_are_empty_until_stage_1():
    assert CFG["fitted"] == {"scalers": None, "widths": None, "thresholds": None}


def test_missing_section_is_an_error(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("data: {}\n", encoding="utf8")
    with pytest.raises(ValueError, match="missing section"):
        load_config(p)
