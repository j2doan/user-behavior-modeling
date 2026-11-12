"""v4 models: out-of-fold stacking, EASE, and a transformer that sees the reader's genre history.

Same interface as v2/v3. Builds on v3 (which builds on v2).
  Rating  : 5-fold out-of-fold stacking (GBDT trained on every training row, not a 20% slice)
            + explicit zero-star modelling (smoothed user/item zero rates, nonzero-only bias model).
  Read    : v3 ranker trained on 5 out-of-fold rounds (~10x more labelled pairs) + EASE and cosine features.
  Category: v3's linear model blended with roberta-base, which sees the reader's genre history as a text prefix.
"""
import importlib.util
import os
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

_spec = importlib.util.spec_from_file_location(
    "v3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v3", "gbdt_stack_roberta.py"))
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
v2 = v3.v2

SEED = 0
GBDT = dict(v3.GBDT, max_iter=1000)


def _folds(n, k):
    return np.array_split(np.random.default_rng(SEED).permutation(n), k)


##################################################
# Rating                                         #
##################################################
BIAS = dict(K=0, lam_u=4.0, lam_i=16.0, lam_g=10.0, iters=100, tol=1e-7)   # tuned in v2 benchmark
MF = dict(K=5, lam_u=4.0, lam_i=4.0, lam_g=20.0, iters=100, tol=1e-7)
RATING_PARAMS = dict(bias=BIAS, mf=MF, folds=5, smooth=5.0)
RATING_GRID = [RATING_PARAMS, dict(RATING_PARAMS, mf=None), dict(RATING_PARAMS, smooth=20.0)]
RATING_PARAMS = dict(RATING_PARAMS, mf=None)  # best on benchmark validation


def _stats(rows, keyIdx, smooth, z0):
    g = defaultdict(list)
    for t in rows:
        g[t[keyIdx]].append(t[2])
    out = {}
    for k, rs in g.items():
        rs = np.array(rs, dtype=float)
        nz = rs[rs > 0]
        out[k] = (len(rs), rs.mean(), rs.std(), (rs == 0).mean(),
                  ((rs == 0).sum() + smooth * z0) / (len(rs) + smooth), nz.mean() if len(nz) else np.nan)
    return out


def _rating_base(train, bias, mf, smooth):
    z0 = np.mean([r == 0 for *_, r in train])
    return dict(bias=v2.fit_rating(train, **bias),
                nz=v2.fit_rating([t for t in train if t[2] > 0], **bias),
                mf=mf and v2.fit_rating(train, **mf),
                U=_stats(train, 0, smooth, z0), I=_stats(train, 1, smooth, z0), z0=z0)


def _rating_features(m, pairs):
    cols = [v2.predict_rating(m["bias"], pairs), v2.predict_rating(m["nz"], pairs)]
    if m["mf"]:
        cols.append(v2.predict_rating(m["mf"], pairs))
    none = (0, np.nan, np.nan, np.nan, m["z0"], np.nan)
    stats = [m["U"].get(u, none) + m["I"].get(b, none) for u, b in pairs]
    return np.hstack([np.array(cols, dtype=float).T, np.array(stats, dtype=float)])


def fit_rating(train, bias, mf, folds, smooth):
    X, y = [], []
    for idx in _folds(len(train), folds):
        held = set(idx.tolist())
        rest = [t for n, t in enumerate(train) if n not in held]
        part = [train[n] for n in idx]
        X.append(_rating_features(_rating_base(rest, bias, mf, smooth), [(u, b) for u, b, _ in part]))
        y += [r for *_, r in part]
    gbdt = HistGradientBoostingRegressor(**GBDT).fit(np.vstack(X), y)
    return dict(base=_rating_base(train, bias, mf, smooth), gbdt=gbdt)


def predict_rating(m, pairs):
    return np.clip(m["gbdt"].predict(_rating_features(m["base"], pairs)), 0, 5).tolist()


##################################################
# Read                                           #
##################################################
# EASE (Steck 2019): closed-form item-item linear model, B = argmin ||X - XB||^2 + lam||B||^2, diag(B)=0.
# score(u,b) = sum over books j the user read of B[j, b]. A close relative of SLIM.

READ_PARAMS = dict(bpr=v3.READ_PARAMS["bpr"], folds=5, ease_lam=500.0)
READ_GRID = [dict(READ_PARAMS, ease_lam=lam) for lam in (100.0, 500.0, 2000.0)]
READ_PARAMS = dict(READ_PARAMS, ease_lam=100.0)  # best on benchmark validation


def _read_base(train, bpr, ease_lam):
    m = v3._read_base(train, bpr)
    items = v2._index(b for _, b, _ in train)
    users = v2._index(u for u, _, _ in train)
    pairs = {(users[u], items[b]) for u, b, _ in train}
    X = sp.csr_matrix((np.ones(len(pairs)), tuple(np.array(list(pairs)).T)), shape=(len(users), len(items)))
    G = (X.T @ X).toarray() + ease_lam * np.eye(len(items))
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0.0)
    m.update(easeB=B, easeItems=items)
    return m


def _read_features(m, pairs):
    base = v3._read_features(m, pairs)
    extra = []
    for u, b in pairs:
        Ub, Iu = m["usersPerItem"].get(b, set()), m["itemsPerUser"].get(u, set())
        cos = max((len(Ub & m["usersPerItem"][b2]) / np.sqrt(len(Ub) * len(m["usersPerItem"][b2]))
                   for b2 in Iu if b2 != b and Ub), default=0.0)
        bi = m["easeItems"].get(b)
        js = [m["easeItems"][j] for j in Iu if j != b]
        ease = m["easeB"][js, bi].sum() if bi is not None and js else 0.0
        extra.append((cos, ease))
    return np.hstack([base, np.array(extra, dtype=float)])


def fit_read(train, bpr, folds, ease_lam):
    rng = np.random.default_rng(SEED)
    X, y = [], []
    for idx in _folds(len(train), folds):
        held = set(idx.tolist())
        rest = [t for n, t in enumerate(train) if n not in held]
        pos = [(train[n][0], train[n][1]) for n in idx]
        pairs = pos + v3._sample_negatives(pos, train, rng)
        X.append(_read_features(_read_base(rest, bpr, ease_lam), pairs))
        y += [1] * len(pos) + [0] * len(pos)
    clf = HistGradientBoostingClassifier(**GBDT).fit(np.vstack(X), y)
    return dict(base=_read_base(train, bpr, ease_lam), clf=clf)


def predict_read(m, pairs):
    score = m["clf"].predict_proba(_read_features(m["base"], pairs))[:, 1]
    byUser = defaultdict(list)
    for n, (u, _) in enumerate(pairs):
        byUser[u].append((score[n], n))
    out = [0] * len(pairs)
    for cands in byUser.values():
        cands.sort(reverse=True)
        for _, n in cands[:(len(cands) + 1) // 2]:
            out[n] = 1
    return out


##################################################
# Category                                       #
##################################################
# v3 linear model (word TF-IDF + user history -> LogReg) blended with a fine-tuned transformer that also
# sees the reviewer's genre history as a text prefix ("reader history: fantasy 3, mystery 1 | <review>").
# Training texts use leave-one-out history (never the review's own label), like the linear model.
# Trained transformers are cached in-process, so a grid over w_tf costs one fine-tune, not one per entry.

GENRE_NAMES = ["children", "comics", "fantasy", "mystery", "young adult"]
CATEGORY_PARAMS = dict(C=8.0, w_user=1.0, char=True, transformer="roberta-base", user_prefix=True,
                       epochs=2, max_len=256, batch=16, lr=2e-5, w_tf=0.7)
CATEGORY_GRID = [dict(CATEGORY_PARAMS, w_tf=w) for w in (0.5, 0.7, 0.85, 1.0)] \
    + [dict(CATEGORY_PARAMS, user_prefix=False, w_tf=w) for w in (0.5, 0.7)]
CATEGORY_PARAMS = dict(CATEGORY_PARAMS, w_tf=0.5)  # best on benchmark validation
_TF_CACHE, _LIN_CACHE = {}, {}


def _prefixed(reviews, counts):
    out = []
    for d, c in zip(reviews, counts):
        hist = ", ".join(f"{GENRE_NAMES[g]} {int(n)}" for g, n in enumerate(c) if n > 0) or "none"
        out.append(f"reader history: {hist} | {d['review_text']}")
    return out


def _counts(m, reviews):
    return np.array([m["userCounts"].get(d["user_id"], np.zeros(v2.N_GENRES)) for d in reviews])


def fit_category(train, C, w_user, char, transformer, user_prefix, epochs, max_len, batch, lr, w_tf):
    data = hash(tuple(d["review_id"] for d in train))
    if (C, w_user, char, data) not in _LIN_CACHE:
        _LIN_CACHE[(C, w_user, char, data)] = v3.fit_category(
            train, C=C, w_user=w_user, char=char, transformer=None,
            epochs=epochs, max_len=max_len, batch=batch, lr=lr, w_tf=0.0)
    lin = _LIN_CACHE[(C, w_user, char, data)]
    m = dict(lin=lin, w_tf=w_tf, user_prefix=user_prefix, max_len=max_len, tf=None)
    if transformer:
        y = np.array([d["genreID"] for d in train])
        key = (transformer, user_prefix, epochs, max_len, batch, lr, data)
        if key not in _TF_CACHE:
            loo = _counts(lin, train) - np.eye(v2.N_GENRES)[y]
            texts = _prefixed(train, loo) if user_prefix else [d["review_text"] for d in train]
            _TF_CACHE[key] = v3._fit_transformer(texts, y, transformer, epochs, max_len, batch, lr)
        m["tf"] = _TF_CACHE[key]
    return m


def predict_category(m, reviews):
    lin, counts = m["lin"], _counts(m["lin"], reviews)
    p = lin["clf"].predict_proba(v3._linear_X(lin, reviews, counts))
    if m["tf"]:
        texts = _prefixed(reviews, counts) if m["user_prefix"] else [d["review_text"] for d in reviews]
        p = (1 - m["w_tf"]) * p + m["w_tf"] * v3._transformer_proba(*m["tf"], texts, m["max_len"])
    return [int(k) for k in p.argmax(1)]
