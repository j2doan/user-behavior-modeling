"""Train v0 on the full data and write outputs/v0-output/predictions_*.csv.

    python run.py
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bias_jaccard_tfidf.py"), run_name="__main__")
