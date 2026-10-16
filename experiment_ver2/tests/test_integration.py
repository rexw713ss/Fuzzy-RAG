import json
from zipfile import ZipFile

import numpy as np
import pytest

from experiment_ver2.data import load_dataset
from experiment_ver2.demo import create_demo
from experiment_ver2.metrics import arrays, paired
from experiment_ver2.package_cache import package
from experiment_ver2.run import BUDGETS, evaluate_dataset, main


@pytest.fixture
def cache(tmp_path):
    root = create_demo(tmp_path / "cache", n_clusters=30)
    return root, load_dataset(root, "synthetic")


def test_test_labels_and_costs_cannot_change_fits_or_decisions(cache):
    _, dataset = cache
    report, model, decisions = evaluate_dataset(dataset, n_boot=100)
    dataset.test.f1[:] = 1 - dataset.test.f1
    dataset.test.cost[:] *= 10
    dataset.test.em[:] = 1 - dataset.test.em
    changed, new_model, new_decisions = evaluate_dataset(dataset, n_boot=100)
    assert decisions == new_decisions
    assert model == new_model
    for q in BUDGETS:
        for method in report["budgets"][str(q)]["methods"]:
            assert report["budgets"][str(q)]["methods"][method]["fit"] == changed["budgets"][str(q)]["methods"][method]["fit"]
    for row in report["budgets"].values():
        for result in row["methods"].values():
            assert result["counts"]["A2"] <= int(row["quota"] * len(dataset.test) + 1e-12)


def test_random_reference_uses_selected_count_not_test_optimum(cache):
    _, data = cache
    a = np.ones(len(data.test), dtype=int)
    a[:3] = 2
    p = 3 / len(a)
    result = paired(data.test, a, random_fraction=p, n_boot=100)
    expected = ((1 - p) * data.test.f1[:, 1] + p * data.test.f1[:, 2]).mean()
    assert result["reference"]["means"]["F1"] == pytest.approx(expected)
    assert result["reference"]["A2_fraction"] == p


def test_fractional_action_is_rejected(cache):
    _, data = cache
    with pytest.raises(ValueError):
        arrays(data.test, np.full(len(data.test), 1.9))


def test_minimal_bundle_roundtrip(cache, tmp_path):
    root, _ = cache
    archive = tmp_path / "bundle.zip"
    manifest = package(root, archive)
    assert len(manifest["files"]) == 4
    with ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == set(manifest["files"]) | {"CACHE_MANIFEST.json"}
        destination = tmp_path / "restored"
        bundle.extractall(destination)
    assert len(load_dataset(destination, "synthetic").test) == 18
    with pytest.raises(ValueError, match="overwrite"):
        package(root, archive)


def test_cli_missing_data_reports_failure_without_benchmark(tmp_path):
    output = tmp_path / "missing-report"
    assert main(["--root", str(tmp_path / "absent"), "--out", str(output)]) == 2
    assert json.loads((output / "failure.json").read_text())["benchmark_completed"] is False
    assert not (output / "REPORT.md").exists()


def test_cli_check_only_and_no_overwrite(cache, tmp_path):
    root, _ = cache
    output = tmp_path / "audit"
    assert main(["--root", str(root), "--out", str(output), "--check-only"]) == 0
    assert (output / "data_audit.json").is_file()
    with pytest.raises(SystemExit):
        main(["--root", str(root), "--out", str(output), "--check-only"])


def test_cli_refuses_outputs_inside_original_data(cache):
    root, _ = cache
    with pytest.raises(SystemExit):
        main(["--root", str(root), "--out", str(root / "new-run"), "--check-only"])
