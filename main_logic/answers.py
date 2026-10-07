"""Does a passage contain a gold answer? (DPR's has_answer; used for Recall@k, spec v0.3 7.4.)

Follows DPR's string-match rule so retrieval recall is comparable with published NQ numbers:
Unicode NFD normalization, DPR's SimpleTokenizer (runs of letters/digits/marks, or any single
other visible character), lowercase, then a contiguous token-sequence match. Only the passage
text is searched, not its title, as in DPR.
"""

import unicodedata

import regex

_TOKEN = regex.compile(r"([\p{L}\p{N}\p{M}]+)|([^\p{Z}\p{C}])",
                       flags=regex.IGNORECASE | regex.UNICODE | regex.MULTILINE)


def tokens(s):
    return [m.group().lower() for m in _TOKEN.finditer(unicodedata.normalize("NFD", s))]


def has_answer(answers, text):
    """True if any answer's token sequence appears contiguously in the text's tokens."""
    t = tokens(text)
    for ans in answers:
        a = tokens(ans)
        n = len(a)
        if n and any(t[i:i + n] == a for i in range(len(t) - n + 1)):
            return True
    return False


def recall_at(hit_flags, ks):
    """{k: fraction of questions with an answer in the top k}. hit_flags: per question, a list
    of booleans in rank order."""
    return {k: sum(any(f[:k]) for f in hit_flags) / len(hit_flags) for k in ks}
