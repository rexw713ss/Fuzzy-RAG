"""config.yaml must equal the constants the code uses (spec v0.3, 7.5).

If a value changes in one place and not the other, a test here fails.
"""

from pathlib import Path

import pytest

from main_logic import (actions, baselines, bm25, calibration, dense, evaluation, fuzzy, routing, signals,
                        splits)
from main_logic.config import load_config
from scripts import datasets, monotonicity_report

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
    ("baselines", "b1_weight_step", baselines.WEIGHT_STEP),
    ("baselines", "b2_t_max", baselines.T_MAX),
    ("calibration", "quantile", calibration.QUANTILE),
    ("calibration", "cluster_size", calibration.CLUSTER_SIZE),
    ("splits", "seed", splits.SEED),
    ("splits", "names", list(splits.SPLITS)),
    ("evaluation", "oracle_delta", evaluation.DELTA),
    ("evaluation", "bootstrap_resamples", evaluation.N_BOOT),
    ("evaluation", "ci_level", evaluation.CI_LEVEL),
    ("evaluation", "bootstrap_seed", evaluation.SEED),
])
def test_config_matches_code(section, key, constant):
    assert CFG[section][key] == constant


def test_evaluation_values_are_the_specs():
    """6.1 fixes delta at 0.02 and 6.3 fixes 1,000 resamples."""
    assert evaluation.DELTA == 0.02
    assert evaluation.N_BOOT == 1000


def test_data_paths_match_the_scripts():
    """The scripts take every path from scripts/datasets.py; it must equal config.yaml."""
    data = CFG["data"]
    assert Path(data["root"]) == datasets.DATA == monotonicity_report.DATA
    for name, prefix in (("nq", ""), ("hotpotqa", "hotpotqa_")):
        assert Path(data[prefix + "corpus"]) == datasets.CORPUS[name]
        assert Path(data[prefix + "embeddings"]) == datasets.EMBEDDINGS[name]
        assert Path(data[prefix + "bm25_index"]) == datasets.BM25_INDEX[name]


def test_fitted_values_are_empty_until_stage_1():
    assert CFG["fitted"] == {"scalers": None, "widths": None, "thresholds": None}


def test_missing_section_is_an_error(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("data: {}\n", encoding="utf8")
    with pytest.raises(ValueError, match="missing section"):
        load_config(p)
