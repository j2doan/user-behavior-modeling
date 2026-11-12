"""Tune v5's stand-alone SGD latent-factor model on the benchmark's validation split (test never touched).
Start at K=1, tune the two regularizers, early stopping, clip. Prints val MSE per setting.

    python tune_sgd.py
"""
import itertools
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "outputs"))
import benchmark as B  # noqa: E402

A = B.load("v5", "v5/ensemble_roberta_large.py")

itr, iva, _ = B.split3(B.R.read_interactions())
pairs, y = [(u, b) for u, b, _ in iva], [r for *_, r in iva]


def run(**p):
    t = time.time()
    m = A.fit_sgd(itr, **p)
    print(f"{B.R.mse(A.predict_sgd(m, pairs), y):.4f}  epochs={m['epochs']}  ({time.time() - t:.0f}s)  {p}", flush=True)


base = dict(max_epochs=60, patience=3)
# round 1 (init 0.01): the factor never left ~0 -- lam_g made no difference; best lr 0.005, lam_b 0.1 -> 1.4607
for lr, lam_b, lam_g in itertools.product((0.005, 0.01), (0.03, 0.1, 0.3), (0.03, 0.1, 0.3)):
    run(**base, K=1, init=0.01, lr=lr, lam_b=lam_b, lam_g=lam_g)
# round 2: larger initial factors so the factor term can actually learn
for K, init, lam_g in itertools.product((1, 3), (0.1, 0.3), (0.03, 0.1, 0.3)):
    run(**base, K=K, init=init, lr=0.005, lam_b=0.1, lam_g=lam_g)
