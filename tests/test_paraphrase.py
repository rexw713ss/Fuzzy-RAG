"""Tests for the paraphrase-cluster logic (spec v0.3, section 7.1). The gate models themselves
are not loaded here."""

import numpy as np
import pytest

from main_logic import paraphrase as pp

ENT = 2                                   # deberta-large-mnli: 0 contra, 1 neutral, 2 entail
YES = np.array([0.01, 0.04, 0.95])
NO = np.array([0.10, 0.60, 0.30])


def run(candidates, fwd=None, bwd=None, cos=None, q="who wrote hallelujah", answers=("Leonard Cohen",)):
    n = len(candidates)
    return pp.screen(q, list(answers), candidates, fwd or [YES] * n, bwd or [YES] * n,
                     cos or [0.9] * n, ENT)


def test_sample_is_seeded_and_prefix_stable():
    a = pp.sample(1000, 300)
    assert a == pp.sample(1000, 300)
    assert pp.sample(1000, 20) == a[:20]          # the pilot is the start of the full sample
    assert len(set(a)) == 300 and all(0 <= i < 1000 for i in a)
    assert pp.sample(1000, 20, seed=7) != a[:20]
    with pytest.raises(ValueError):
        pp.sample(5, 6)


def test_messages_contain_the_prompt_and_question():
    m = pp.messages("who wrote hallelujah")
    assert len(m) == 1 and m[0]["role"] == "user"
    assert pp.PROMPT in m[0]["content"] and "who wrote hallelujah" in m[0]["content"]


@pytest.mark.parametrize("reply, expected", [
    ("Who wrote Hallelujah?", "Who wrote Hallelujah?"),
    ('  "Who wrote Hallelujah?"  ', "Who wrote Hallelujah?"),
    ("Rewritten question: Who wrote Hallelujah?", "Who wrote Hallelujah?"),
    ("\n\nWho wrote Hallelujah?\n(I kept the meaning.)", "Who wrote Hallelujah?"),
    ("", ""),
    (None, ""),
    ('""', ""),
])
def test_clean(reply, expected):
    assert pp.clean(reply) == expected


def test_all_gates_pass():
    r = run(["Who is the writer of Hallelujah?"])
    assert r[0]["passed"] and r[0]["reason"] is None and r[0]["entails"]


def test_nli_must_hold_in_both_directions():
    assert run(["x one"], fwd=[NO])[0]["reason"] == "nli"
    assert run(["x one"], bwd=[NO])[0]["reason"] == "nli"


def test_cosine_gate_boundary():
    assert run(["x one"], cos=[pp.COSINE_MIN])[0]["passed"]
    assert run(["x one"], cos=[pp.COSINE_MIN - 1e-6])[0]["reason"] == "cosine"


def test_answer_leak_gate():
    r = run(["Did Leonard Cohen write Hallelujah?"])
    assert r[0]["leak"] and r[0]["reason"] == "answer"


def test_answer_already_in_the_original_is_not_a_leak():
    q = "which is older, the Eiffel Tower or Big Ben?"
    r = run(["Which is older: Big Ben or the Eiffel Tower?"], q=q, answers=["Big Ben"])
    assert not r[0]["leak"] and r[0]["passed"]


def test_empty_identical_and_repeated_candidates_fail_first():
    r = run(["", "WHO  wrote hallelujah", "Who is the writer of Hallelujah?",
             "who is the writer of hallelujah?"])
    assert [x["reason"] for x in r] == ["empty", "duplicate", None, "duplicate"]


def test_case_and_punctuation_only_changes_are_duplicates():
    r = run(["Who wrote Hallelujah?", "who wrote hallelujah.", "Who wrote 'Hallelujah'"])
    assert [x["reason"] for x in r] == ["duplicate"] * 3
    assert pp.normalized("Who  wrote, Hallelujah?") == "who wrote hallelujah"


def test_select_takes_the_first_passing_in_order():
    r = run(["a b", "c d", "e f", "g h"], fwd=[NO, YES, YES, YES])
    assert pp.select(r) == ["c d", "e f"]
    assert pp.select(run(["a b"], cos=[0.1])) == []


def test_entailment_index():
    assert pp.entailment_index({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}) == 2
    assert pp.entailment_index({"0": "entailment", "1": "neutral"}) == 0
    with pytest.raises(ValueError):
        pp.entailment_index({0: "yes", 1: "no"})
