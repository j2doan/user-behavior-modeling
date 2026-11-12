"""v5 models: built on v4. Same interface as v2-v4.

  Rating  : v4's out-of-fold GBDT stack + an SGD latent-factor model
            (small K, separate bias/factor regularizers, early stopping on a held-out slice, clipping).
  Read    : v4's ranker + user-user Jaccard (is u like the other readers of b?) and co-read counts.
  Category: weighted average of 4 members' class probabilities, weights tuned on validation:
            linear (v3/v4 TF-IDF + user history), roberta-base, xlm-roberta-base (multilingual, for the
            ~6.5% non-English reviews), roberta-large (512 tokens). Transformers see the reader's genre
            history as a text prefix (v4). Member probabilities are cached on disk, so a crash only loses
            the member being trained and the weight grid costs nothing.
"""
import gc
import hashlib
import importlib.util
import os
import random
import time
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("v4", os.path.join(_HERE, "..", "v4", "oof_ease_roberta.py"))
v4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4)
v3, v2 = v4.v3, v4.v2

SEED = 0


def _log(msg):
    print(f"      [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


##################################################
# Rating                                         #
##################################################
# f(u,b) = alpha + beta_u + beta_b + gamma_u . gamma_b, SGD with separate regularizers for betas and gammas.
# Early stopping: train on 90%, track MSE on the other 10%, keep the best epoch count, refit on 100%.

SGD = dict(K=1, lr=0.005, lam_b=0.1, lam_g=0.1, init=0.1, max_epochs=60, patience=3)  # tuned by tune_sgd.py (val 1.4606)
RATING_PARAMS = dict(bias=v4.BIAS, sgd=dict(SGD, K=3), folds=5, smooth=5.0)  # K=3 won the benchmark (tie)
RATING_GRID = [dict(RATING_PARAMS, sgd=dict(SGD, K=k)) for k in (1, 3)]


def _sgd(train, K, lr, lam_b, lam_g, init, epochs, held=None, patience=0):
    rng = random.Random(SEED)
    nrng = np.random.default_rng(SEED)
    users, items = {}, {}
    data = [(users.setdefault(u, len(users)), items.setdefault(b, len(items)), r) for u, b, r in train]
    m = dict(users=users, items=items, alpha=np.mean([r for *_, r in data]), bu=np.zeros(len(users)),
             bi=np.zeros(len(items)), gu=init * nrng.standard_normal((len(users), K)),
             gi=init * nrng.standard_normal((len(items), K)))
    a, bu, bi, gu, gi = m["alpha"], m["bu"], m["bi"], m["gu"], m["gi"]
    hist = []
    for _ in range(epochs):
        rng.shuffle(data)
        for u, b, r in data:
            err = r - (a + bu[u] + bi[b] + gu[u] @ gi[b])
            a += lr * err
            bu[u] += lr * (err - lam_b * bu[u])
            bi[b] += lr * (err - lam_b * bi[b])
            g_u = gu[u].copy()
            gu[u] += lr * (err * gi[b] - lam_g * g_u)
            gi[b] += lr * (err * g_u - lam_g * gi[b])
        m["alpha"] = a
        if held is not None:
            hist.append(np.mean((np.array(predict_sgd(m, [(u, b) for u, b, _ in held])) -
                                 [r for *_, r in held]) ** 2))
            if len(hist) - 1 - int(np.argmin(hist)) >= patience:
                break
    return m, hist


def fit_sgd(train, K, lr, lam_b, lam_g, init, max_epochs, patience):
    rest, held = v3._split(train, 0.1)
    _, hist = _sgd(rest, K, lr, lam_b, lam_g, init, max_epochs, held=held, patience=patience)
    best = int(np.argmin(hist)) + 1
    m, _ = _sgd(train, K, lr, lam_b, lam_g, init, best)
    m.update(epochs=best, held_mse=min(hist))
    return m


def predict_sgd(m, pairs):
    out = []
    for u, b in pairs:
        ui, bj = m["users"].get(u), m["items"].get(b)
        p = m["alpha"] + (m["bu"][ui] if ui is not None else 0.0) + (m["bi"][bj] if bj is not None else 0.0)
        if ui is not None and bj is not None:
            p += m["gu"][ui] @ m["gi"][bj]
        out.append(min(5.0, max(0.0, p)))
    return out


def _rating_base(train, bias, sgd, smooth):
    m = v4._rating_base(train, bias, None, smooth)
    m["sgd"] = fit_sgd(train, **sgd)
    return m


def _rating_features(m, pairs):
    return np.hstack([v4._rating_features(m, pairs), np.array(predict_sgd(m["sgd"], pairs))[:, None]])


def fit_rating(train, bias, sgd, folds, smooth):
    X, y = [], []
    for idx in v4._folds(len(train), folds):
        held = set(idx.tolist())
        rest = [t for n, t in enumerate(train) if n not in held]
        part = [train[n] for n in idx]
        X.append(_rating_features(_rating_base(rest, bias, sgd, smooth), [(u, b) for u, b, _ in part]))
        y += [r for *_, r in part]
    gbdt = HistGradientBoostingRegressor(**v4.GBDT).fit(np.vstack(X), y)
    return dict(base=_rating_base(train, bias, sgd, smooth), gbdt=gbdt)


def predict_rating(m, pairs):
    return np.clip(m["gbdt"].predict(_rating_features(m["base"], pairs)), 0, 5).tolist()


##################################################
# Read                                           #
##################################################
# Extra features on top of v4: user-user Jaccard between u's books and each other reader of b (max, mean),
# and the co-read count sum_{b2 in I_u} |U_b & U_b2|.

READ_PARAMS = dict(v4.READ_PARAMS, ease_lam=100.0)  # v4's best
READ_GRID = [READ_PARAMS]


def _read_features(m, pairs):
    extra = []
    for u, b in pairs:
        Ub, Iu = m["usersPerItem"].get(b, set()), m["itemsPerUser"].get(u, set())
        uu = [v2._jaccard(Iu, m["itemsPerUser"][v]) for v in Ub if v != u]
        co = sum(len(Ub & m["usersPerItem"][b2]) for b2 in Iu if b2 != b)
        extra.append((max(uu, default=0.0), np.mean(uu) if uu else 0.0, co))
    return np.hstack([v4._read_features(m, pairs), np.array(extra, dtype=float)])


def fit_read(train, bpr, folds, ease_lam):
    rng = np.random.default_rng(SEED)
    X, y = [], []
    for idx in v4._folds(len(train), folds):
        held = set(idx.tolist())
        rest = [t for n, t in enumerate(train) if n not in held]
        pos = [(train[n][0], train[n][1]) for n in idx]
        pairs = pos + v3._sample_negatives(pos, train, rng)
        X.append(_read_features(v4._read_base(rest, bpr, ease_lam), pairs))
        y += [1] * len(pos) + [0] * len(pos)
    clf = HistGradientBoostingClassifier(**v4.GBDT).fit(np.vstack(X), y)
    return dict(base=v4._read_base(train, bpr, ease_lam), clf=clf)


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
# Members train lazily inside predict_category (the first time their probabilities for that
# (train set, review set) are needed), save the probabilities to CACHE, then free the GPU.
# shortcut: a trained transformer is not kept, so predicting a *new* review set retrains it; fine for the
# benchmark and the runner (one review set per fit), keep the model object if that ever changes.

MEMBERS = {  # name -> transformer config (None = linear model)
    "linear": None,
    "roberta-base": dict(max_len=256, batch=16, accum=1, lr=2e-5, freeze_emb=False),
    "xlm-roberta-base": dict(max_len=256, batch=16, accum=1, lr=2e-5, freeze_emb=False),
    "roberta-large": dict(max_len=512, batch=4, accum=4, lr=1e-5, freeze_emb=True),
}
NAMES = list(MEMBERS)
EPOCHS = 2
COLLAPSE_ACC = 0.5  # epoch-1 training accuracy below this = collapsed (majority class is 36%)
CACHE = os.path.join(_HERE, "cache")

CATEGORY_PARAMS = dict(w=(0.4, 0.1, 0.0, 0.5))  # best on benchmark validation
CATEGORY_GRID = [dict(w=(0.25, 0.25, 0.25, 0.25))] + [dict(w=(a / 10, b / 10, c / 10, (10 - a - b - c) / 10))
                                     for a in range(11) for b in range(11 - a) for c in range(11 - a - b)]


def _key(rows):
    return hashlib.md5("|".join(d["review_id"] for d in rows).encode()).hexdigest()[:12]


def _user_counts(train):
    uc = defaultdict(lambda: np.zeros(v2.N_GENRES))
    for d in train:
        uc[d["user_id"]][d["genreID"]] += 1
    return dict(userCounts=dict(uc))


def _member_proba(name, train, reviews):
    path = os.path.join(CACHE, f"{name}_{_key(train)}_{_key(reviews)}.npy")
    if os.path.exists(path):
        return np.load(path)
    _log(f"training {name} on {len(train)} reviews")
    if MEMBERS[name] is None:
        lin = v3.fit_category(train, C=8.0, w_user=1.0, char=True, transformer=None,
                              epochs=0, max_len=0, batch=0, lr=0.0, w_tf=0.0)
        p = lin["clf"].predict_proba(v3._linear_X(lin, reviews, v4._counts(lin, reviews)))
    else:
        cfg = MEMBERS[name]
        y = np.array([d["genreID"] for d in train])
        uc = _user_counts(train)
        texts = v4._prefixed(train, v4._counts(uc, train) - np.eye(v2.N_GENRES)[y])  # leave-one-out history
        tok, model = _fit_transformer(texts, y, name, **cfg)
        p = _transformer_proba(tok, model, v4._prefixed(reviews, v4._counts(uc, reviews)), cfg["max_len"])
        del tok, model
        _free_gpu()
    os.makedirs(CACHE, exist_ok=True)
    np.save(path, p)
    _log(f"{name} done")
    return p


def fit_category(train, w):
    return dict(train=train, w=w)


def predict_category(m, reviews):
    p = sum(w * _member_proba(n, m["train"], reviews) for n, w in zip(NAMES, m["w"]) if w > 0)
    return [int(k) for k in p.argmax(1)]


# ---------- transformer fine-tuning ----------
def _free_gpu():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def _batches(lengths, batch, rng):
    """Shuffled batches of similar-length texts (less padding): sort within random chunks of 50 batches."""
    idx = rng.permutation(len(lengths))
    out = []
    for s in range(0, len(idx), batch * 50):
        chunk = sorted(idx[s:s + batch * 50], key=lambda k: lengths[k])
        out += [chunk[k:k + batch] for k in range(0, len(chunk), batch)]
    return [out[k] for k in rng.permutation(len(out))]


def _fit_transformer(texts, y, name, max_len, batch, accum, lr, freeze_emb):
    for seed in (SEED, SEED + 1):
        out = _train_once(texts, y, name, max_len, batch, accum, lr, freeze_emb, seed)
        if out:
            return out
        _free_gpu()
    raise RuntimeError(f"{name} collapsed with two seeds")


def _train_once(texts, y, name, max_len, batch, accum, lr, freeze_emb, seed):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=v2.N_GENRES).to(dev)
    if accum > 1:
        model.gradient_checkpointing_enable()
    if freeze_emb:  # saves the word-embedding gradients + Adam state (~0.8 GB for roberta-large)
        model.get_input_embeddings().weight.requires_grad_(False)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    lengths = [len(t) for t in texts]
    per_epoch = -(-len(_batches(lengths, batch, rng)) // accum)
    steps, warm = EPOCHS * per_epoch, max(1, int(0.06 * EPOCHS * per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))
    check = rng.choice(len(texts), min(2000, len(texts)), replace=False)
    for ep in range(EPOCHS):
        model.train()
        bs = _batches(lengths, batch, rng)
        for k, idx in enumerate(bs):
            enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16):
                loss = model(**enc, labels=torch.as_tensor(y[idx], device=dev)).loss / accum
            loss.backward()
            if (k + 1) % accum == 0 or k + 1 == len(bs):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
            if k % 2000 == 0:
                _log(f"{name} epoch {ep + 1} batch {k}/{len(bs)} loss {loss.item() * accum:.3f}")
        acc = np.mean(_transformer_proba(tok, model, [texts[i] for i in check], max_len).argmax(1) == y[check])
        _log(f"{name} epoch {ep + 1} done, accuracy on 2k training reviews {acc:.3f}")
        if ep == 0 and acc < COLLAPSE_ACC:
            _log(f"{name} collapsed (seed {seed}), retrying")
            return None
    model.eval()
    return tok, model


def _transformer_proba(tok, model, texts, max_len, batch=32):
    import torch
    dev = next(model.parameters()).device
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), v2.N_GENRES))
    model.eval()
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            idx = order[s:s + batch]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                out[idx] = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
    return out
