"""Tests for DPR-style answer matching (spec v0.3, section 7.4 Recall@5)."""

import pytest

from main_logic import answers as an


def test_tokenizer_splits_words_and_single_symbols():
    assert an.tokens("U.S. Route 66, 1926!") == ["u", ".", "s", ".", "route", "66", ",", "1926", "!"]


def test_match_is_case_insensitive_and_contiguous():
    text = "Apollo 17 left the Moon on 14 December 1972."
    assert an.has_answer(["december 1972"], text)
    assert not an.has_answer(["1972 December"], text)          # order matters


def test_match_is_on_whole_tokens_not_substrings():
    assert not an.has_answer(["Moo"], "Apollo 17 left the Moon")
    assert an.has_answer(["Moon"], "Apollo 17 left the Moon")


def test_unicode_is_normalized():
    assert an.has_answer(["Ahärôn"], "Aaron ( or ; Ahärôn ) is a prophet")
    # composed vs decomposed forms of the same letters must match
    assert an.has_answer(["Beyoncé"], "Beyoncé released Lemonade")


def test_any_of_several_answers_and_empty_answers():
    assert an.has_answer(["Bob Russell", "Bobby Scott"], "lyrics by Bobby Scott and others")
    assert not an.has_answer(["", "  "], "anything")


def test_recall_at_k():
    flags = [[False, True, False], [False, False, False], [True, False, False]]
    assert an.recall_at(flags, [1, 2, 3]) == pytest.approx({1: 1 / 3, 2: 2 / 3, 3: 2 / 3})
