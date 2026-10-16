import numpy as np
import pytest

from experiment_ver2.policies import choose_on_validation, quota_actions


def test_quota_does_not_force_harmful_escalation():
    a = quota_actions([.5, -.2, 0, .1], ["a", "b", "c", "d"], 1)
    assert a.tolist() == [2, 1, 1, 2]


def test_quota_floor_and_stable_ties():
    a = quota_actions([1] * 7, [str(i) for i in range(7)], .25)
    assert (a == 2).sum() == 1
    permutation = np.array([3, 2, 5, 6, 4, 0, 1])
    b = quota_actions([1] * 7, [str(i) for i in permutation], .25)
    assert np.array_equal(a[permutation], b)


def test_validation_fallback_can_win():
    f1 = np.array([[.1, .8, .7], [.1, .8, .7]])
    fit = choose_on_validation([("rescue", [2, 2]), ("fallback", [1, 1])], f1, np.ones((2, 3)))
    assert fit["candidate"] == "fallback"


def test_ties_prefer_fewer_calls():
    fit = choose_on_validation([("rescue", [2, 2]), ("fallback", [1, 1])],
                               np.ones((2, 3)), np.ones((2, 3)))
    assert fit["candidate"] == "fallback"


@pytest.mark.parametrize("fraction", [-.1, 1.1, float("nan")])
def test_invalid_quota(fraction):
    with pytest.raises(ValueError):
        quota_actions([1], ["q"], fraction)


def test_uncertainty_baseline_can_use_nonpositive_score():
    assert quota_actions([0, 0], ["a", "b"], 1, positive_only=False).tolist() == [2, 2]
