"""v1 models. Same interface as v2 (see v2/als_bpr_linear.py).
  Rating  : latent-factor model trained by SGD.
  Read    : Bayesian personalized ranking (v2's numpy BPR), per-user top half.
  Category: TF-IDF (1-3 grams, 10k features, stop words) -> logistic regression.
"""
import importlib.util
import os
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_spec = importlib.util.spec_from_file_location(
    "v2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v2", "als_bpr_linear.py"))
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

SEED = 42


##################################################
# Rating prediction                              #
##################################################
# f(u,b) = alpha + beta_u + beta_b + gamma_u . gamma_b, one SGD step per rating:
#   theta += lr * (error * grad - reg * theta)
# One regularizer for everything; unseen user or book -> alpha; no clipping.

RATING_PARAMS = dict(nf=20, lr=0.01, reg=0.05, epochs=10)
RATING_GRID = [dict(nf=nf, lr=lr, reg=reg, epochs=10)
               for nf in (10, 20) for lr in (0.005, 0.01) for reg in (0.01, 0.05, 0.1)]
RATING_PARAMS = dict(RATING_PARAMS, nf=10)  # best on benchmark validation


def fit_rating(train, nf, lr, reg, epochs):
    random.seed(SEED)
    np.random.seed(SEED)
    users, items = {}, {}
    data = [(users.setdefault(u, len(users)), items.setdefault(b, len(items)), r) for u, b, r in train]
    alpha = np.mean([r for *_, r in data])
    bu, bi = np.zeros(len(users)), np.zeros(len(items))
    gu = 0.01 * np.random.randn(len(users), nf)
    gi = 0.01 * np.random.randn(len(items), nf)
    for _ in range(epochs):
        random.shuffle(data)
        for u, b, r in data:
            err = r - (alpha + bu[u] + bi[b] + np.dot(gu[u], gi[b]))
            alpha += lr * err
            bu[u] += lr * (err - reg * bu[u])
            bi[b] += lr * (err - reg * bi[b])
            g_u, g_b = gu[u].copy(), gi[b].copy()
            gu[u] += lr * (err * g_b - reg * g_u)
            gi[b] += lr * (err * g_u - reg * g_b)
    return dict(users=users, items=items, alpha=alpha, bu=bu, bi=bi, gu=gu, gi=gi)


def predict_rating(m, pairs):
    out = []
    for u, b in pairs:
        ui, bj = m["users"].get(u), m["items"].get(b)
        if ui is None or bj is None:
            out.append(float(m["alpha"]))
        else:
            out.append(float(m["alpha"] + m["bu"][ui] + m["bi"][bj] + np.dot(m["gu"][ui], m["gi"][bj])))
    return out


##################################################
# Read prediction                                #
##################################################
# Bayesian personalized ranking: learn gamma_u . gamma_b (+ item bias) so a read book outranks an unread one.
# Within each user, the top-scoring half of their candidate books is predicted 1.

READ_PARAMS = dict(K=10, lr=0.05, lam=1e-4, epochs=20)
READ_GRID = [dict(READ_PARAMS, K=k, epochs=e) for k in (10, 20, 50) for e in (20, 50)]


def fit_read(train, K, lr, lam, epochs):
    return v2.fit_read(train, w_pop=0.0, w_jac=0.0, w_bpr=1.0, K=K, lr=lr, lam=lam, epochs=epochs)


predict_read = v2.predict_read


##################################################
# Category prediction                            #
##################################################

CATEGORY_PARAMS = dict(C=1.0)
CATEGORY_GRID = [dict(C=c) for c in (1.0, 5.0, 10.0, 50.0)]


def fit_category(train, C):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=10000, stop_words="english",
                                  sublinear_tf=True, smooth_idf=True)),
        ("clf", LogisticRegression(max_iter=5000, random_state=SEED, C=C)),
    ])
    return model.fit([d["review_text"] for d in train], [d["genreID"] for d in train])


def predict_category(m, reviews):
    return [int(k) for k in m.predict([d["review_text"] for d in reviews])]
