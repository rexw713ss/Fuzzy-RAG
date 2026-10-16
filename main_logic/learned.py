"""Query-only learned router (spec v0.3, section 5 reference point; section 2.7).

Sees only the question text, never the retrieval signals, so comparing it with B1-B4 answers RQ2
(do retrieval-state signals add value beyond the query?). Section 2.7 names the features: length,
entity count, multi-hop cue words.

Choices the spec leaves open (Claude, Oct 11, 2026 -- DECISIONS.md section 14):
  - features: word count, capitalized words after the first (an entity-count proxy; no NER
    model), number present, the question word (who/what/when/where/which/how/why/other), count of
    multi-hop cue words; standardized on the training split;
  - model: multinomial logistic regression, L2 penalty 0.01, full-batch gradient descent
    (2,000 steps, rate 0.5, zero start), so it is deterministic; numpy only;
  - labels: the oracle action of each training instance (section 6.1 supplies them);
  - routing: the score is the expected action, (P(A1) * 0.5 + P(A2) * 1.0), in [0, 1], and two
    thresholds are chosen on validation with the section 3.4 objective, budget and tie-break,
    exactly as for B1-B4.
"""

import numpy as np

WH = ("who", "what", "when", "where", "which", "how", "why")
CUES = frozenset({"and", "both", "or", "than", "same", "also", "older", "younger", "earlier",
                  "later", "first", "more", "less"})
L2 = 0.01
STEPS = 2000
RATE = 0.5


def features(question):
    """Raw query-only features of one question (before standardization)."""
    words = question.split()
    lower = [w.strip("?,.!\"'():;").lower() for w in words]
    wh = next((w for w in lower if w in WH), None)
    return np.array([
        len(words),
        sum(w[:1].isupper() for w in words[1:]),
        float(any(ch.isdigit() for ch in question)),
        *[float(wh == x) for x in WH], float(wh is None),
        sum(w in CUES for w in lower),
    ], dtype=float)


def feature_matrix(questions):
    return np.array([features(q) for q in questions])


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit(questions, labels, n_classes=3):
    """Train on (question, oracle action index) pairs. Returns the model as a dict."""
    X = feature_matrix(questions)
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = np.hstack([(X - mean) / sd, np.ones((len(X), 1))])
    Y = np.eye(n_classes)[np.asarray(labels, dtype=int)]
    W = np.zeros((Xs.shape[1], n_classes))
    for _ in range(STEPS):
        grad = Xs.T @ (_softmax(Xs @ W) - Y) / len(Xs)
        grad[:-1] += L2 * W[:-1]                    # no penalty on the bias row
        W -= RATE * grad
    return {"mean": mean.tolist(), "sd": sd.tolist(), "W": W.tolist()}


def probabilities(model, questions):
    X = feature_matrix(questions)
    Xs = np.hstack([(X - np.array(model["mean"])) / np.array(model["sd"]), np.ones((len(X), 1))])
    return _softmax(Xs @ np.array(model["W"]))


def scores(model, questions):
    """Expected action scaled to [0, 1]: sum_j P(action j) * j / (k - 1).

    Three classes (v0.3): P(A1) * 0.5 + P(A2) * 1.0. Two classes (v0.4: A1, A2): P(A2).
    """
    p = probabilities(model, questions)
    k = p.shape[1]
    return p @ (np.arange(k) / (k - 1))
