"""Strict train/val/test benchmark on the labelled training data (the test files have no labels).

Splits (fixed seed): 200k interactions and 100k labelled reviews -> 80% train / 10% val / 10% test.
  Rating  : MSE on held-out interactions
  Read    : held-out interactions + one sampled never-read book each (per-user balanced), accuracy
  Category: accuracy on held-out reviews
Protocol per model & task: fit every *_GRID entry on train, score on val, keep the best;
refit the best on train+val and score ONCE on test. Results -> outputs/benchmark/<model>/.

    python benchmark.py --models baseline v0 v1 v2 v3 v4 v5 [--tasks rating read category] [--no-tune]
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark")
FILES = dict(v0="bias_jaccard_tfidf", v1="sgd_bpr_tfidf", v2="als_bpr_linear", v3="gbdt_stack_roberta",
             v4="oof_ease_roberta", v5="ensemble_roberta_large")
SEED = 158


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, os.path.join(HERE, "v2"))  # the runner imports its sibling model file
R = load("runner_shared", "v2/run.py")  # I/O, baselines, metrics, read negative sampling


def split3(rows):
    rows = list(rows)
    random.Random(SEED).shuffle(rows)
    n = len(rows) // 10
    return rows[:-2 * n], rows[-2 * n:-n], rows[-n:]


# ---------- model adapters: task -> (fit(train, **p), predict(model, X), params, grid) ----------
def baseline_model():
    return {
        "rating": (lambda tr: tr, R.baseline_rating, {}, None),
        "read": (lambda tr: tr, R.baseline_read, {}, None),
        "category": (lambda tr: None, lambda m, revs: R.baseline_category(revs), {}, None),
    }


def v0_model():
    O = load("v0", "v0/bias_jaccard_tfidf.py")

    def per(tr):
        pu, pi = defaultdict(list), defaultdict(list)
        for u, b, r in tr:
            pu[u].append((b, r))
            pi[b].append((u, r))
        return pu, pi

    def fit_rating(tr):
        pu, pi = per(tr)
        with contextlib.redirect_stdout(io.StringIO()):  # goodModel prints every iteration
            return O.goodModel(tr, pu, pi, O.getGlobalAverage([r for *_, r in tr]),
                               {u: 0 for u in pu}, {b: 0 for b in pi})

    def predict_rating(m, pairs):
        a, bu, bi = m
        return [a + bu.get(u, 0) + bi.get(b, 0) for u, b in pairs]

    def predict_read(m, pairs):
        pu, pi = m
        return [O.jaccardThresh(u, b, pi, pu) for u, b in pairs]

    def fit_category(tr):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        p = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=10000, stop_words="english",
                                                sublinear_tf=True, smooth_idf=True)),
                      ("clf", LogisticRegression(max_iter=5000, random_state=42, C=1))])
        return p.fit([d["review_text"] for d in tr], [d["genreID"] for d in tr])

    return {
        "rating": (fit_rating, predict_rating, {}, None),
        "read": (per, predict_read, {}, None),
        "category": (fit_category, lambda m, revs: [int(x) for x in m.predict([d["review_text"] for d in revs])], {}, None),
    }


def versioned_model(name):
    A = load(name, f"{name}/{FILES[name]}.py")
    return {t: (getattr(A, f"fit_{t}"), getattr(A, f"predict_{t}"), getattr(A, f"{t.upper()}_PARAMS"),
                getattr(A, f"{t.upper()}_GRID")) for t in ("rating", "read", "category")}


def v3_rating_grid(grid):
    """v3 stacks v2's rating models: plug in the best tuned v2 bias (K=0) and latent-factor (K>0) params."""
    path = os.path.join(OUT, "v2", "metrics.json")
    if not os.path.exists(path):
        return grid
    entries = json.load(open(path))["rating"]["grid"]
    ok = [e for e in entries if "val" in e]
    bias = min((e for e in ok if e["params"]["K"] == 0), key=lambda e: e["val"])["params"]
    mf = min((e for e in ok if e["params"]["K"] > 0), key=lambda e: e["val"])["params"]
    return [dict(p, bias=bias, mf=mf) for p in grid]


# ---------- evaluation ----------
def evaluate(fit, predict, params, grid, tune, train, val, test, Xval, yval, Xtest, ytest, metric, better):
    tried = []
    for p in (grid if tune and grid else [params]):
        t = time.time()
        try:
            score = metric(predict(fit(train, **p), Xval), yval)
            tried.append(dict(params=p, val=score, sec=round(time.time() - t)))
        except Exception as e:  # keep the rest of the grid going (e.g. a model download failing)
            tried.append(dict(params=p, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-800:]))
        print(f"    {tried[-1].get('val', tried[-1].get('error'))}  ({round(time.time() - t)}s)  {p}", flush=True)
    good = [e for e in tried if "val" in e]
    best = (min if better == "min" else max)(good, key=lambda e: e["val"])
    t = time.time()
    pred = predict(fit(train + val, **best["params"]), Xtest)
    return dict(best_params=best["params"], val=best["val"], test=metric(pred, ytest),
                refit_sec=round(time.time() - t), grid=tried), pred


def save_preds(folder, name, header, rows):
    with open(os.path.join(folder, name), "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=["rating", "read", "category"])
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()

    inter = R.read_interactions()
    itr, iva, ite = split3(inter)
    rv_pairs, rv_y = [(u, b) for u, b, _ in iva], [r for *_, r in iva]
    rt_pairs, rt_y = [(u, b) for u, b, _ in ite], [r for *_, r in ite]
    dv_pairs, dv_y = R.read_validation(iva, inter)
    dt_pairs, dt_y = R.read_validation(ite, inter)
    reviews = R.read_reviews("train_Category.json.gz")
    ctr, cva, cte = split3(reviews)

    for name in args.models:
        model = {"baseline": baseline_model, "v0": v0_model}.get(name, lambda: versioned_model(name))()
        folder = os.path.join(OUT, name)
        os.makedirs(folder, exist_ok=True)
        mpath = os.path.join(folder, "metrics.json")
        metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
        for task in args.tasks:
            fit, predict, params, grid = model[task]
            if name == "v3" and task == "rating":
                params, grid = v3_rating_grid([params])[0], v3_rating_grid(grid)
            print(f"\n=== {name} / {task} ===", flush=True)
            if task == "rating":
                res, pred = evaluate(fit, predict, params, grid, not args.no_tune, itr, iva, ite,
                                     rv_pairs, rv_y, rt_pairs, rt_y, R.mse, "min")
                save_preds(folder, "predictions_Rating.csv", "userID,bookID,prediction,label",
                           [(u, b, p, y) for (u, b), p, y in zip(rt_pairs, pred, rt_y)])
            elif task == "read":
                res, pred = evaluate(fit, predict, params, grid, not args.no_tune, itr, iva, ite,
                                     dv_pairs, dv_y, dt_pairs, dt_y, R.accuracy, "max")
                save_preds(folder, "predictions_Read.csv", "userID,bookID,prediction,label",
                           [(u, b, p, y) for (u, b), p, y in zip(dt_pairs, pred, dt_y)])
            else:
                res, pred = evaluate(fit, predict, params, grid, not args.no_tune, ctr, cva, cte,
                                     cva, [d["genreID"] for d in cva], cte, [d["genreID"] for d in cte],
                                     R.accuracy, "max")
                save_preds(folder, "predictions_Category.csv", "userID,reviewID,prediction,label",
                           [(d["user_id"], d["review_id"], p, d["genreID"]) for d, p in zip(cte, pred)])
            print(f"  -> best val {res['val']:.4f}   TEST {res['test']:.4f}   {res['best_params']}", flush=True)
            metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}  # other tasks may run in parallel
            metrics[task] = res
            json.dump(metrics, open(mpath, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
