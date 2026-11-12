"""Runner: validates each task against the baselines, then trains on the full data and writes predictions.
Same file in every model folder (v1-v5) except the model import below.

    python run.py                      # validate all tasks vs. baselines, then write outputs/<version>-output/
    python run.py --tasks rating       # any subset of: rating read category
    python run.py --tune               # also score every entry of *_GRID on validation
    python run.py --no-write           # validation only (skip full-data refit + prediction files)

Validation (the test files have no labels, so we hold out training data):
    rating   : random 10k of the 200k interactions, MSE
    read     : same 10k as positives + one sampled unread book per positive (per-user balanced), accuracy
    category : random 10k of the 100k labeled reviews, accuracy
"""
import argparse
import ast
import gzip
import os
import random
import time
from collections import defaultdict

import oof_ease_roberta as A

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "outputs", os.path.basename(HERE) + "-output")  # outputs/v1-output, ...
SEED = 0
N_VALID = 10000


# ---------- I/O ----------
def read_interactions():
    with gzip.open(os.path.join(DATA, "train_Interactions.csv.gz"), "rt") as f:
        next(f)
        return [(u, b, int(r)) for u, b, r in (l.strip().split(",") for l in f)]


def read_reviews(name):
    with gzip.open(os.path.join(DATA, name), "rt") as f:
        return [ast.literal_eval(l) for l in f]


def read_pairs(name):
    with open(os.path.join(DATA, name)) as f:
        next(f)
        return [tuple(l.strip().split(",")) for l in f]


def write_predictions(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"  wrote outputs/{os.path.basename(HERE)}/{name} ({len(rows)} rows)")


def split(rows, n_valid):
    rows = list(rows)
    random.Random(SEED).shuffle(rows)
    return rows[:-n_valid], rows[-n_valid:]


# ---------- baselines (same logic as data/baselines.py) ----------
def baseline_rating(train, pairs):
    userRatings = defaultdict(list)
    for u, _, r in train:
        userRatings[u].append(r)
    glob = sum(r for _, _, r in train) / len(train)
    return [sum(userRatings[u]) / len(userRatings[u]) if u in userRatings else glob for u, _ in pairs]


def baseline_read(train, pairs):
    count = defaultdict(int)
    for _, b, _ in train:
        count[b] += 1
    return1, seen = set(), 0
    for c, b in sorted(((c, b) for b, c in count.items()), reverse=True):
        seen += c
        return1.add(b)
        if seen > len(train) / 2:
            break
    return [int(b in return1) for _, b in pairs]


def baseline_category(reviews):
    out = []
    for d in reviews:
        words, cat = d["review_text"].lower(), 2
        for word, c in (("children", 0), ("comic", 1), ("fantasy", 2), ("mystery", 3), ("love", 4)):
            if word in words:
                cat = c
        out.append(cat)
    return out


# ---------- metrics ----------
def mse(pred, y):
    return sum((p - t) ** 2 for p, t in zip(pred, y)) / len(y)


def accuracy(pred, y):
    return sum(p == t for p, t in zip(pred, y)) / len(y)


def read_validation(valid, all_rows):
    """Positives = held-out (u,b); negatives = one book u never read per positive."""
    readBy = defaultdict(set)
    for u, b, _ in all_rows:
        readBy[u].add(b)
    books = sorted({b for _, b, _ in all_rows})
    rng = random.Random(SEED)
    pairs, labels = [], []
    for u, b, _ in valid:
        neg = rng.choice(books)
        while neg in readBy[u]:
            neg = rng.choice(books)
        pairs += [(u, b), (u, neg)]
        labels += [1, 0]
    return pairs, labels


# ---------- tasks ----------
def run_task(name, fit, predict, params, grid, train, vpairs, vlabels, base_pred, metric, tune):
    print(f"\n=== {name} ===")
    print(f"  baseline   valid {metric.__name__} = {metric(base_pred, vlabels):.4f}")
    for p in (grid if tune else [params]):
        t = time.time()
        score = metric(predict(fit(train, **p), vpairs), vlabels)
        print(f"  model      valid {metric.__name__} = {score:.4f}  ({time.time() - t:.0f}s)  {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["rating", "read", "category"],
                    choices=["rating", "read", "category"])
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    if {"rating", "read"} & set(args.tasks):
        inter = read_interactions()
        itrain, ivalid = split(inter, N_VALID)

    if "rating" in args.tasks:
        vpairs, vlabels = [(u, b) for u, b, _ in ivalid], [r for _, _, r in ivalid]
        run_task("Rating", A.fit_rating, A.predict_rating, A.RATING_PARAMS, A.RATING_GRID,
                 itrain, vpairs, vlabels, baseline_rating(itrain, vpairs), mse, args.tune)
        if not args.no_write:
            pairs = read_pairs("pairs_Rating.csv")
            preds = A.predict_rating(A.fit_rating(inter, **A.RATING_PARAMS), pairs)
            write_predictions("predictions_Rating.csv", "userID,bookID,prediction",
                              [(u, b, p) for (u, b), p in zip(pairs, preds)])

    if "read" in args.tasks:
        vpairs, vlabels = read_validation(ivalid, inter)
        run_task("Read", A.fit_read, A.predict_read, A.READ_PARAMS, A.READ_GRID,
                 itrain, vpairs, vlabels, baseline_read(itrain, vpairs), accuracy, args.tune)
        if not args.no_write:
            pairs = read_pairs("pairs_Read.csv")
            preds = A.predict_read(A.fit_read(inter, **A.READ_PARAMS), pairs)
            write_predictions("predictions_Read.csv", "userID,bookID,prediction",
                              [(u, b, p) for (u, b), p in zip(pairs, preds)])

    if "category" in args.tasks:
        reviews = read_reviews("train_Category.json.gz")
        ctrain, cvalid = split(reviews, N_VALID)
        run_task("Category", A.fit_category, A.predict_category, A.CATEGORY_PARAMS, A.CATEGORY_GRID,
                 ctrain, cvalid, [d["genreID"] for d in cvalid], baseline_category(cvalid), accuracy, args.tune)
        if not args.no_write:
            test = read_reviews("test_Category.json.gz")  # same order as pairs_Category.csv
            preds = A.predict_category(A.fit_category(reviews, **A.CATEGORY_PARAMS), test)
            write_predictions("predictions_Category.csv", "userID,reviewID,prediction",
                              [(d["user_id"], d["review_id"], p) for d, p in zip(test, preds)])


if __name__ == "__main__":
    main()
