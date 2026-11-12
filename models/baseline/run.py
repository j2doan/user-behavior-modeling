"""Write outputs/baseline-output/predictions_*.csv with the baseline rules.

    python run.py
"""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines.py"), run_name="__main__")
