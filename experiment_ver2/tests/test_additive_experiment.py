"""Leakage, literal cluster inference and unchanged-reference checks."""

from copy import deepcopy
import json

import numpy as np
import pytest

from experiment_ver2 import additive_models
from experiment_ver2.data import Dataset, Split
from experiment_ver2.run import RIDGE_ALPHAS
from experiment_ver2.run_additive import select_and_predict, compare, subgroups
from experiment_ver2.robustness.run import verify_reference
from experiment_ver2.robustness.statistics import cluster_counts, interval


def _dataset(n_actions=2):
    rng = np.random.default_rng(216)
    actions = ("A1", "A2") if n_actions == 2 else ("A0", "A1", "A2")

    def split(name, sizes):
        features, quality, cost, keys, questions = [], [], [], [], []
        for cluster, size in enumerate(sizes):
            centre = rng.uniform(.05, .95, 4)
            for variant in range(size):
                x = np.clip(centre + rng.normal(0, .03, 4), 0, 1)
                gain = .35 if centre[0] < .5 else -.2
                features.append(x)
                quality.append(([.2] if n_actions == 3 else []) + [.45, .45 + gain])
                cost.append(([.8] if n_actions == 3 else []) + [1., 2.])
                keys.append(("toy", f"{name}-{cluster}", variant))
                questions.append(f"{name} question {cluster} variant {variant}")
        return Split(np.array(features), np.array(quality), np.array(cost),
                     np.zeros((len(keys), n_actions)), tuple(keys), tuple(questions), actions=actions)

    return Dataset("toy", split("train", [3]*8 + [2, 1]), split("val", [3]*5 + [1]),
                   split("test", [3, 1, 2, 3]), np.array([.2, .21, .24, .25]), actions=actions)


@pytest.mark.parametrize("n_actions", [2, 3])
def test_test_outcome_perturbation_cannot_change_any_model_selection_score_or_action(n_actions):
    original = _dataset(n_actions)
    altered = deepcopy(original)
    altered.test.f1 = 1 - altered.test.f1
    altered.test.cost = altered.test.cost * 100 + 37
    altered.test.em = 1 - altered.test.em
    models_a, fits_a, scores_a, actions_a = select_and_predict(original)
    models_b, fits_b, scores_b, actions_b = select_and_predict(altered)
    assert models_a == models_b
    assert fits_a == fits_b
    for name in scores_a:
        np.testing.assert_array_equal(scores_a[name], scores_b[name])
        np.testing.assert_array_equal(actions_a[name], actions_b[name])
    assert json.dumps(models_a, allow_nan=False)


def test_selection_never_reads_test_outcomes_even_if_absent():
    original = _dataset()
    expected_models, expected_fits, expected_scores, expected_actions = select_and_predict(original)
    original.test.f1 = None
    original.test.cost = None
    original.test.em = None
    actual_models, actual_fits, actual_scores, actual_actions = select_and_predict(original)
    assert actual_models == expected_models
    assert actual_fits == expected_fits
    for name in expected_scores:
        np.testing.assert_array_equal(actual_scores[name], expected_scores[name])
        np.testing.assert_array_equal(actual_actions[name], expected_actions[name])


def test_training_is_cluster_equal_and_validation_choice_matches_literal_candidates():
    data = _dataset()
    candidates, fits, _, assignments = select_and_predict(data)
    groups = data.train.groups
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    for method, variants in candidates.items():
        assert set(variants) == {f"alpha={a}" for a in RIDGE_ALPHAS}
        for candidate, model in variants.items():
            assert model["training_weight"] == pytest.approx(10)
            np.testing.assert_allclose(model["global_quality"],
                                       np.average(data.train.f1, weights=weights, axis=0))
            quality, _ = additive_models.predict(model, data.val.X)
            score = quality[:, 1] - quality[:, 0]
            # Candidate policy is evaluated only on VAL with positive-gain
            # eligibility and the fixed quota; no TEST outcome participates.
            from experiment_ver2.policies import quota_actions
            actions = quota_actions(score, data.val.questions, .25, base=0, escalation=1)
            actual = next(c for c in fits[method]["all_candidates"] if c["candidate"] == candidate)
            assert actual["F1"] == pytest.approx(data.val.f1[np.arange(len(data.val)), actions].mean())
            assert actual["A2_count"] == int(np.sum(actions == 1))
        expected = min(fits[method]["all_candidates"],
                       key=lambda p: (-p["F1"], p["A2_count"], p["cost"], p["candidate"]))
        assert fits[method]["candidate"] == expected["candidate"]
        assert np.sum(assignments[method] == 1) <= int(.25 * len(data.test))


def _inference_split(n_actions):
    quality = np.array([[0, 1], [.1, .9], [.4, .6], [.8, .2], [.3, .7], [.5, .5]])
    labels = ("A1", "A2")
    if n_actions == 3:
        quality = np.column_stack((np.zeros(6), quality))
        labels = ("A0", "A1", "A2")
    return Split(np.zeros((6, 4)), quality, np.ones((6, n_actions)), np.zeros_like(quality),
                 (("d", "a", 0), ("d", "a", 1), ("d", "a", 2),
                  ("d", "b", 0), ("d", "c", 0), ("d", "c", 1)),
                 tuple("abcdef"), actions=labels)


@pytest.mark.parametrize("n_actions", [2, 3])
def test_comparison_matches_literal_duplicated_cluster_formula_and_family_adjustment(n_actions):
    split = _inference_split(n_actions)
    base, escalation = (split.actions.index(x) for x in ("A1", "A2"))
    actions = np.array([escalation, base, base, base, escalation, base])
    refs = {"ridge": np.array([base, escalation, base, base, base, escalation]),
            "fuzzy-calibrated": np.full(6, base)}
    observed = compare(split, actions, refs, n_boot=100, seed=312)
    counts, inverse = cluster_counts(split.groups, 100, 312)
    samples = {name: [] for name in (*refs, "same_count_random")}
    for mult in counts[:, inverse]:
        rows = np.repeat(np.arange(len(split)), mult)
        q = split.f1[rows]
        assigned = actions[rows]
        selected = q[np.arange(len(rows)), assigned].mean()
        for name, reference in refs.items():
            other = reference[rows]
            samples[name].append(selected - q[np.arange(len(rows)), other].mean())
        p = np.mean(assigned == escalation)
        samples["same_count_random"].append(selected - ((1-p)*q[:, base] + p*q[:, escalation]).mean())
    quality = split.f1[np.arange(len(split)), actions]
    for name, boot in samples.items():
        if name == "same_count_random":
            p = np.mean(actions == escalation)
            reference = (1-p)*split.f1[:, base] + p*split.f1[:, escalation]
        else:
            reference = split.f1[np.arange(len(split)), refs[name]]
        expected_point = (quality-reference).mean()
        result = observed["comparisons"][name]
        assert result["difference"] == pytest.approx(expected_point)
        np.testing.assert_allclose(result["percentile_95"], np.quantile(boot, [.025, .975]), atol=1e-14)
        np.testing.assert_allclose(result["bonferroni_percentile"],
                                   np.quantile(boot, [.025/12, 1-.025/12]), atol=1e-14)
        assert result["bonferroni_marginal_confidence"] == pytest.approx(1-.05/12)
    assert observed["family_size"] == 12
    assert len(observed["comparisons"]) == 3
    assert "recomputed inside each cluster resample" in observed["estimand"]


def test_frozen_subset_summary_neither_reranks_nor_drops_original_queries():
    split = _inference_split(2)
    actions = np.array([1, 0, 0, 1, 1, 0])
    observed = subgroups(split, actions)
    original_rows = [0, 3, 4]
    assert observed["original_only"]["queries"] == 3
    assert observed["original_only"]["A2_count"] == 3
    assert observed["original_only"]["F1"] == pytest.approx(split.f1[original_rows, 1].mean())
    assert observed["complete_clusters_only"]["queries"] == 3
    assert observed["complete_clusters_only"]["A2_count"] == 1


def test_reference_verification_rejects_changed_candidate_action_or_quality(tmp_path):
    data = _dataset()
    actions = np.zeros(len(data.test), dtype=int)
    fit = {"ridge": {"candidate": "alpha=1.0"}}
    assignments = {"ridge": actions}
    report = {"budgets": {"0.25": {"methods": {"ridge": {
        "fit": fit["ridge"], "F1": {"mean": float(data.test.f1[:, 0].mean())}}}}}}
    decisions = [{"method": "ridge", "cluster_id": k[1], "variant_id": k[2], "quota": .25, "action": "A1"}
                 for k in data.test.keys]
    (tmp_path/"toy_report.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path/"toy_decisions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in decisions), encoding="utf-8")
    assert verify_reference(data, fit, assignments, tmp_path)["all_six_families_primary_candidates_actions_F1_match"]
    with pytest.raises(ValueError, match="candidate changed"):
        verify_reference(data, {"ridge": {"candidate": "alpha=10.0"}}, assignments, tmp_path)
    changed = actions.copy()
    changed[0] = 1
    with pytest.raises(ValueError, match="decisions changed"):
        verify_reference(data, fit, {"ridge": changed}, tmp_path)
    data.test.f1[:, 0] = 0
    with pytest.raises(ValueError, match="F1 changed"):
        verify_reference(data, fit, assignments, tmp_path)
