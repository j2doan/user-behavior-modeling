"""v3 models: GBDT stacking and a fine-tuned transformer. Same interface as v2, so the runner is shared.

    fit_rating / predict_rating, fit_read / predict_read, fit_category / predict_category
    *_PARAMS = chosen config, *_GRID = candidates for `run.py --tune`

Reuses v2's models (loaded from ../v2/als_bpr_linear.py) as base learners and stacks a
second-stage model on top.
"""
import importlib.util
import os
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

_spec = importlib.util.spec_from_file_location(
    "v2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v2", "als_bpr_linear.py"))
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)

SEED = 0
GBDT = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=SEED)


def _split(rows, frac):
    """Random (rest, held-out) split of a list."""
    idx = np.random.default_rng(SEED).permutation(len(rows))
    n = int(len(rows) * frac)
    return [rows[k] for k in idx[n:]], [rows[k] for k in idx[:n]]


##################################################
# Rating prediction: stacked ensemble            #
##################################################
# Base learners (fit on 90%): v2 bias model, v2 latent-factor model (ALS), user/item rating
# statistics (count, mean, std, fraction of 0-star ratings -- 0s are ~5% of ratings and dominate MSE).
# A gradient-boosted tree regressor learns how to combine them on the held-out 10%, then the base
# learners are refit on 100% and the GBDT is applied to their outputs.

RATING_PARAMS = dict(bias=dict(v2.RATING_PARAMS), mf=dict(v2.RATING_PARAMS, K=5, lam_g=20.0), blend_frac=0.1)
RATING_GRID = [RATING_PARAMS, dict(RATING_PARAMS, blend_frac=0.2)]
RATING_PARAMS = dict(RATING_PARAMS, mf=dict(RATING_PARAMS["mf"], lam_i=4.0), blend_frac=0.2)  # best on benchmark validation

_NO_STATS = (0, np.nan, np.nan, np.nan)


def _rating_stats(train):
    byU, byI = defaultdict(list), defaultdict(list)
    for u, b, r in train:
        byU[u].append(r)
        byI[b].append(r)
    st = lambda rs: (len(rs), np.mean(rs), np.std(rs), np.mean(np.array(rs) == 0))
    return {u: st(rs) for u, rs in byU.items()}, {b: st(rs) for b, rs in byI.items()}


def _rating_base(train, bias, mf):
    return (v2.fit_rating(train, **bias), v2.fit_rating(train, **mf)) + _rating_stats(train)


def _rating_features(base, pairs):
    mBias, mMF, statU, statI = base
    pb, pm = v2.predict_rating(mBias, pairs), v2.predict_rating(mMF, pairs)
    return np.array([(x, y) + statU.get(u, _NO_STATS) + statI.get(b, _NO_STATS)
                     for (u, b), x, y in zip(pairs, pb, pm)], dtype=float)


def fit_rating(train, bias, mf, blend_frac):
    fitPart, blend = _split(train, blend_frac)
    X = _rating_features(_rating_base(fitPart, bias, mf), [(u, b) for u, b, _ in blend])
    gbdt = HistGradientBoostingRegressor(**GBDT).fit(X, [r for _, _, r in blend])
    return dict(base=_rating_base(train, bias, mf), gbdt=gbdt)


def predict_rating(m, pairs):
    return np.clip(m["gbdt"].predict(_rating_features(m["base"], pairs)), 0, 5).tolist()


##################################################
# Read prediction: learned ranker + per-user top half
##################################################
# Positives = held-out 10% of interactions, negatives = one uniformly sampled unread book each
# (same construction as the test file). Features come from the other 90%:
#   popularity, user activity, max/mean/sum item-item Jaccard, BPR score (v2),
#   and how often the book appears in the query file itself (transductive: test positives follow
#   popularity, sampled negatives don't).
# A GBDT classifier scores pairs; per user, the top-scoring half is predicted 1.

READ_PARAMS = dict(bpr=dict(K=10, lr=0.05, lam=1e-4, epochs=20), holdout_frac=0.1)
READ_GRID = [READ_PARAMS, dict(READ_PARAMS, bpr=None)]


def _read_base(train, bpr):
    m = v2.fit_read(train, w_pop=0.0, w_jac=0.0, w_bpr=1.0 if bpr else 0.0,
                    **(bpr or dict(K=1, lr=0.0, lam=0.0, epochs=0)))
    m["n"] = len(train)
    return m


def _read_features(m, pairs):
    inQuery = Counter(b for _, b in pairs)
    rows = []
    for u, b in pairs:
        Ub, Iu = m["usersPerItem"].get(b, set()), m["itemsPerUser"].get(u, set())
        sims = [v2._jaccard(Ub, m["usersPerItem"][b2]) for b2 in Iu if b2 != b]
        row = (len(Ub) / m["n"] * 1e4, len(Iu), max(sims, default=0.0),
               np.mean(sims) if sims else 0.0, sum(sims), inQuery[b] / len(pairs) * 1e4)
        if m["w_bpr"]:  # column only exists when BPR is on (an all-NaN column breaks the GBDT binning)
            bpr = np.nan
            if u in m["users"] and b in m["items"]:
                gu, gi, bi = m["bpr"]
                bpr = bi[m["items"][b]] + gu[m["users"][u]] @ gi[m["items"][b]]
            row += (bpr,)
        rows.append(row)
    return np.array(rows, dtype=float)


def _sample_negatives(pos, allRows, rng):
    readBy = defaultdict(set)
    for u, b, _ in allRows:
        readBy[u].add(b)
    books = sorted({b for _, b, _ in allRows})
    neg = []
    for u, _ in pos:
        b = books[rng.integers(len(books))]
        while b in readBy[u]:
            b = books[rng.integers(len(books))]
        neg.append((u, b))
    return neg


def fit_read(train, bpr, holdout_frac):
    rest, held = _split(train, holdout_frac)
    pos = [(u, b) for u, b, _ in held]
    pairs = pos + _sample_negatives(pos, train, np.random.default_rng(SEED))
    X = _read_features(_read_base(rest, bpr), pairs)
    clf = HistGradientBoostingClassifier(**GBDT).fit(X, [1] * len(pos) + [0] * len(pos))
    return dict(base=_read_base(train, bpr), clf=clf)


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
# Category prediction: linear ensemble + transformer
##################################################
# 1) Linear: word TF-IDF (1-2 grams) + character TF-IDF (2-5 grams, robust to typos/names)
#    + v2's leave-one-out user genre history -> logistic regression (probabilities).
# 2) Optional fine-tuned transformer (e.g. "distilroberta-base", "roberta-base") on the review text,
#    GPU + bf16. Downloads the pretrained weights on first use.
# Final = (1 - w_tf) * p_linear + w_tf * p_transformer.

CATEGORY_PARAMS = dict(C=4.0, w_user=1.0, char=True, transformer=None,
                       epochs=2, max_len=256, batch=16, lr=3e-5, w_tf=0.5)
CATEGORY_GRID = (
    [dict(CATEGORY_PARAMS, C=c, char=ch) for c in (2.0, 8.0) for ch in (False, True)]
    + [dict(CATEGORY_PARAMS, transformer=t) for t in ("distilroberta-base", "roberta-base")]
)
CATEGORY_PARAMS = dict(CATEGORY_PARAMS, transformer="roberta-base")  # best on benchmark validation

N_GENRES = v2.N_GENRES


def _linear_X(m, reviews, counts):
    texts = [d["review_text"] for d in reviews]
    blocks = [v.transform(texts) for v in m["vecs"]]
    if m["w_user"]:
        blocks.append(v2._user_features(counts, m["w_user"]))
    return sp.hstack(blocks).tocsr()


def _fit_transformer(texts, y, name, epochs, max_len, batch, lr):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=N_GENRES).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * -(-len(texts) // batch)
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    g = torch.Generator().manual_seed(SEED)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(texts), generator=g).split(batch):
            enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss = model(**enc, labels=torch.as_tensor(y[idx.numpy()], device=dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    model.eval()
    return tok, model


def _transformer_proba(tok, model, texts, max_len, batch=64):
    import torch
    dev = next(model.parameters()).device
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
    return np.vstack(out)


def fit_category(train, C, w_user, char, transformer, epochs, max_len, batch, lr, w_tf):
    texts = [d["review_text"] for d in train]
    y = np.array([d["genreID"] for d in train])
    vecs = [TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)]
    if char:
        vecs.append(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=300000,
                                    sublinear_tf=True))
    for v in vecs:
        v.fit(texts)

    userCounts = defaultdict(lambda: np.zeros(N_GENRES))
    for d in train:
        userCounts[d["user_id"]][d["genreID"]] += 1
    m = dict(vecs=vecs, w_user=w_user, userCounts=dict(userCounts), max_len=max_len, w_tf=w_tf, tf=None)
    loo = np.array([userCounts[d["user_id"]] for d in train]) - np.eye(N_GENRES)[y]
    m["clf"] = LogisticRegression(C=C, max_iter=3000).fit(_linear_X(m, train, loo), y)
    if transformer:
        m["tf"] = _fit_transformer(texts, y, transformer, epochs, max_len, batch, lr)
    return m


def predict_category(m, reviews):
    counts = np.array([m["userCounts"].get(d["user_id"], np.zeros(N_GENRES)) for d in reviews])
    p = m["clf"].predict_proba(_linear_X(m, reviews, counts))
    if m["tf"]:
        p = (1 - m["w_tf"]) * p + m["w_tf"] * _transformer_proba(*m["tf"], [d["review_text"] for d in reviews],
                                                                  m["max_len"])
    return [int(k) for k in p.argmax(1)]
