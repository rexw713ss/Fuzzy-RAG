"""Does a passage contain a gold answer? (DPR's has_answer; used for Recall@k, spec v0.3 7.4.)

Follows DPR's string-match rule so retrieval recall is comparable with published NQ numbers:
Unicode NFD normalization, DPR's SimpleTokenizer (runs of letters/digits/marks, or any single
other visible character), lowercase, then a contiguous token-sequence match. Only the passage
text is searched, not its title, as in DPR.
"""

import string
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


# --- Answer quality: EM and F1 (section 7.4), the standard SQuAD / NQ-open definitions --------

_PUNCT = set(string.punctuation)
_ARTICLES = regex.compile(r"\b(a|an|the)\b")


def normalize_answer(s):
    """The official SQuAD normalization: lower case, ASCII punctuation deleted, articles
    removed, whitespace collapsed."""
    s = "".join(ch for ch in s.lower() if ch not in _PUNCT)
    return " ".join(_ARTICLES.sub(" ", s).split())


def exact_match(prediction, answers):
    """1.0 if the normalized prediction equals any normalized gold answer, else 0.0."""
    p = normalize_answer(prediction)
    return float(any(p == normalize_answer(a) for a in answers))


def _f1(prediction, answer):
    p, g = normalize_answer(prediction).split(), normalize_answer(answer).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for t in p:
        if t in g:
            common[t] = min(p.count(t), g.count(t))
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction, answers):
    """Token F1 between prediction and gold answer, the best over the gold answers."""
    return max(_f1(prediction, a) for a in answers)
