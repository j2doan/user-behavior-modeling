"""v2 models: regularized latent factors (ALS), Jaccard + popularity + BPR, TF-IDF + logistic regression.

Interface used by run.py (shared by v1-v5):
    fit_rating(train, **RATING_PARAMS)     -> model;  predict_rating(model, pairs)     -> [float]
    fit_read(train, **READ_PARAMS)         -> model;  predict_read(model, pairs)       -> [0/1]
    fit_category(train, **CATEGORY_PARAMS) -> model;  predict_category(model, reviews) -> [0..4]
    *_PARAMS = chosen hyperparameters, *_GRID = candidates tried by `run.py --tune`

train/pairs are lists of (user, book, rating) / (user, book); reviews are the dicts from the json files.
"""
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SEED = 0


def _index(keys):
    return {k: n for n, k in enumerate(sorted(set(keys)))}


def _groups(idx, n):
    """Row numbers grouped by index value: groups[k] = rows where idx == k."""
    order = np.argsort(idx, kind="stable")
    return np.split(order, np.cumsum(np.bincount(idx, minlength=n))[:-1])


##################################################
# Rating prediction                              #
##################################################
# Latent-factor model:
#   f(u,i) = alpha + beta_u + beta_i + gamma_u . gamma_i
#   min sum (f(u,i) - R_ui)^2 + lam_u*sum beta_u^2 + lam_i*sum beta_i^2 + lam_g*(sum|gamma_u|^2 + sum|gamma_i|^2)
# Fitted by alternating least squares ("Solution 2"): every block below is a closed-form
# regularized least-squares update. K = 0 is the plain bias model (alpha + beta_u + beta_i),
# but with separate lambdas for users and items.

RATING_PARAMS = dict(K=0, lam_u=4.0, lam_i=4.0, lam_g=10.0, iters=100, tol=1e-7)
RATING_GRID = (
    [dict(RATING_PARAMS, lam_u=lu, lam_i=li) for lu in (1, 2, 4, 8) for li in (2, 4, 8, 16)]
    + [dict(RATING_PARAMS, K=k, lam_g=lg) for k in (2, 5, 10) for lg in (5, 20, 50)]
)
RATING_PARAMS = dict(RATING_PARAMS, lam_i=16.0)  # best on benchmark validation


def _als_block(groups, other_idx, other_g, res, lam):
    """For each row-group, solve (G^T G + lam I) gamma = G^T res (regularized least squares)."""
    K = other_g.shape[1]
    out = np.zeros((len(groups), K))
    reg = lam * np.eye(K)
    for n, rows in enumerate(groups):
        G = other_g[other_idx[rows]]
        out[n] = np.linalg.solve(G.T @ G + reg, G.T @ res[rows])
    return out


def fit_rating(train, K, lam_u, lam_i, lam_g, iters, tol):
    users = _index(u for u, _, _ in train)
    items = _index(b for _, b, _ in train)
    U = np.array([users[u] for u, _, _ in train])
    I = np.array([items[b] for _, b, _ in train])
    R = np.array([r for _, _, r in train], dtype=float)
    nU, nI = len(users), len(items)
    cntU, cntI = np.bincount(U, minlength=nU), np.bincount(I, minlength=nI)

    rng = np.random.default_rng(SEED)
    alpha, bu, bi = R.mean(), np.zeros(nU), np.zeros(nI)
    gu, gi = rng.normal(0, 0.1, (nU, K)), rng.normal(0, 0.1, (nI, K))
    byU, byI = (_groups(U, nU), _groups(I, nI)) if K else (None, None)

    prev = np.inf
    for _ in range(iters):
        dot = (gu[U] * gi[I]).sum(1)
        alpha = (R - bu[U] - bi[I] - dot).mean()
        bu = np.bincount(U, R - alpha - bi[I] - dot, nU) / (lam_u + cntU)
        bi = np.bincount(I, R - alpha - bu[U] - dot, nI) / (lam_i + cntI)
        if K:
            res = R - alpha - bu[U] - bi[I]
            gu = _als_block(byU, I, gi, res, lam_g)
            gi = _als_block(byI, U, gu, res, lam_g)
            dot = (gu[U] * gi[I]).sum(1)
        obj = (((alpha + bu[U] + bi[I] + dot) - R) ** 2).sum() \
            + lam_u * (bu ** 2).sum() + lam_i * (bi ** 2).sum() + lam_g * ((gu ** 2).sum() + (gi ** 2).sum())
        if prev - obj < tol * prev:
            break
        prev = obj
    return dict(alpha=alpha, bu=bu, bi=bi, gu=gu, gi=gi, users=users, items=items)


def predict_rating(m, pairs):
    out = []
    for u, b in pairs:
        p = m["alpha"]
        ui, ii = m["users"].get(u), m["items"].get(b)
        if ui is not None:
            p += m["bu"][ui]
        if ii is not None:
            p += m["bi"][ii]
        if ui is not None and ii is not None:
            p += m["gu"][ui] @ m["gi"][ii]
        out.append(min(5.0, max(0.0, p)))  # ratings live in [0, 5]
    return out


##################################################
# Read prediction                                #
##################################################
# Jaccard similarity on user sets, popularity, and one-class
# recommendation via Bayesian Personalized Ranking:
#   x_uij = beta_i - beta_j + gamma_u.(gamma_i - gamma_j),  maximize ln sigma(x_uij)  (SGD on sampled triples)
# score(u,b) = w_pop*log(1+pop_b) + w_jac*max_jaccard(u,b) + w_bpr*bpr(u,b)
# Decision (optimize relative scores rather than positive vs negative): every user in
# pairs_Read.csv has an even number of pairs (half read, half not), so within each user the
# top-scoring half of their candidate books is predicted 1.

READ_PARAMS = dict(w_pop=1.0, w_jac=30.0, w_bpr=0.0, K=10, lr=0.05, lam=1e-4, epochs=20)
READ_GRID = (
    [dict(READ_PARAMS, w_pop=wp, w_jac=wj) for wp in (0.0, 1.0) for wj in (0.0, 10.0, 30.0, 100.0)]
    + [dict(READ_PARAMS, w_pop=wp, w_jac=wj, w_bpr=1.0) for wp, wj in ((0.0, 0.0), (1.0, 30.0))]
)
READ_PARAMS = dict(READ_PARAMS, w_bpr=1.0)  # best on benchmark validation


def _fit_bpr(U, I, nU, nI, K, lr, lam, epochs, rng, batch=1024):
    gu, gi, bi = rng.normal(0, 0.1, (nU, K)), rng.normal(0, 0.1, (nI, K)), np.zeros(nI)
    for _ in range(epochs):
        perm = rng.permutation(len(U))
        for s in range(0, len(U), batch):
            rows = perm[s:s + batch]
            u, i = U[rows], I[rows]
            # shortcut: j is uniform over all books, so it is occasionally a book u read; negligible at this sparsity
            j = rng.integers(0, nI, len(rows))
            gu_u, d = gu[u].copy(), gi[i] - gi[j]
            g = 1.0 / (1.0 + np.exp(bi[i] - bi[j] + (gu_u * d).sum(1)))  # d/dx ln sigma(x) = 1 - sigma(x)
            np.add.at(gu, u, lr * (g[:, None] * d - lam * gu_u))
            np.add.at(gi, i, lr * (g[:, None] * gu_u - lam * gi[i]))
            np.add.at(gi, j, lr * (-g[:, None] * gu_u - lam * gi[j]))
            np.add.at(bi, i, lr * (g - lam * bi[i]))
            np.add.at(bi, j, lr * (-g - lam * bi[j]))
    return gu, gi, bi


def fit_read(train, w_pop, w_jac, w_bpr, K, lr, lam, epochs):
    usersPerItem, itemsPerUser = defaultdict(set), defaultdict(set)
    for u, b, _ in train:
        usersPerItem[b].add(u)
        itemsPerUser[u].add(b)
    m = dict(usersPerItem=usersPerItem, itemsPerUser=itemsPerUser, w_pop=w_pop, w_jac=w_jac, w_bpr=w_bpr)
    if w_bpr:
        users, items = _index(u for u, _, _ in train), _index(b for _, b, _ in train)
        U = np.array([users[u] for u, _, _ in train])
        I = np.array([items[b] for _, b, _ in train])
        m.update(users=users, items=items, bpr=_fit_bpr(U, I, len(users), len(items), K, lr, lam, epochs,
                                                        np.random.default_rng(SEED)))
    return m


def _jaccard(s1, s2):
    union = len(s1 | s2)
    return len(s1 & s2) / union if union else 0.0


def _read_score(m, u, b):
    users_b = m["usersPerItem"].get(b, set())
    s = m["w_pop"] * np.log1p(len(users_b))
    if m["w_jac"]:
        s += m["w_jac"] * max((_jaccard(users_b, m["usersPerItem"][b2])
                               for b2 in m["itemsPerUser"].get(u, ()) if b2 != b), default=0.0)
    if m["w_bpr"] and u in m["users"] and b in m["items"]:
        gu, gi, bi = m["bpr"]
        s += m["w_bpr"] * (bi[m["items"][b]] + gu[m["users"][u]] @ gi[m["items"][b]])
    return s


def predict_read(m, pairs):
    byUser = defaultdict(list)
    for n, (u, b) in enumerate(pairs):
        byUser[u].append((_read_score(m, u, b), n))
    out = [0] * len(pairs)
    for cands in byUser.values():
        cands.sort(reverse=True)
        for _, n in cands[:(len(cands) + 1) // 2]:
            out[n] = 1
    return out


##################################################
# Category prediction                            #
##################################################
# Bag-of-words -> TF-IDF (sublinear tf, n-grams),
# multiclass logistic regression with C tuned on validation.
# Extra feature: the reviewer's genre history (fraction of their other training reviews in each genre,
# plus a "has history" flag) -- 67% of test reviewers also appear in train_Category.
# Training rows use leave-one-out counts so a review never sees its own label.

CATEGORY_PARAMS = dict(ngram_max=2, min_df=2, C=4.0, w_user=1.0)
CATEGORY_GRID = [dict(CATEGORY_PARAMS, C=c, w_user=w) for c in (1.0, 4.0, 16.0) for w in (0.0, 1.0, 3.0)]
CATEGORY_PARAMS = dict(CATEGORY_PARAMS, C=16.0, w_user=3.0)  # best on benchmark validation

N_GENRES = 5


def _user_features(counts, w):
    tot = counts.sum(1, keepdims=True)
    return sp.csr_matrix(w * np.hstack([counts / np.maximum(tot, 1), (tot > 0).astype(float)]))


def fit_category(train, ngram_max, min_df, C, w_user):
    texts = [d["review_text"] for d in train]
    y = np.array([d["genreID"] for d in train])
    vec = TfidfVectorizer(ngram_range=(1, ngram_max), min_df=min_df, sublinear_tf=True)
    X = vec.fit_transform(texts)

    userCounts = defaultdict(lambda: np.zeros(N_GENRES))
    for d in train:
        userCounts[d["user_id"]][d["genreID"]] += 1
    loo = np.array([userCounts[d["user_id"]] for d in train]) - np.eye(N_GENRES)[y]
    if w_user:
        X = sp.hstack([X, _user_features(loo, w_user)]).tocsr()

    clf = LogisticRegression(C=C, max_iter=2000).fit(X, y)
    return dict(vec=vec, clf=clf, userCounts=dict(userCounts), w_user=w_user)


def predict_category(m, reviews):
    X = m["vec"].transform([d["review_text"] for d in reviews])
    if m["w_user"]:
        counts = np.array([m["userCounts"].get(d["user_id"], np.zeros(N_GENRES)) for d in reviews])
        X = sp.hstack([X, _user_features(counts, m["w_user"])]).tocsr()
    return [int(p) for p in m["clf"].predict(X)]
