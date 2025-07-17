import importlib
import subprocess
import sys

# Import train.py as a module
train = importlib.import_module("src.train")


def test_imports():
    """
    Smoke test: Check that key libraries can be imported.
    """
    import pandas
    import numpy
    import sklearn
    import torch
    import lightgbm
    import joblib
    import yaml
    import matplotlib
    import seaborn
    import plotly
    import mlflow
    import kaggle


def test_seed_reproducibility():
    """
    Smoke test: Asserts that seeding produces reproducible results.
    Fails fast if not reproducible.
    """
    from src.utils.config import load_config
    from src.utils.seed import seed_everything
    import numpy as np

    config = load_config()
    seed_everything(config["seed"])
    arr1 = np.random.rand(5)
    seed_everything(config["seed"])
    arr2 = np.random.rand(5)
    assert (arr1 == arr2).all(), "Random seed is not reproducible!"


def test_train_script_runs():
    """
    Smoke test: Ensure train.py runs without crashing.
    """
    result = subprocess.run([sys.executable, "src/train.py"], capture_output=True)
    assert result.returncode == 0, f"train.py failed: {result.stderr.decode()}"


def test_submit_script_runs():
    """
    Smoke test: Ensure submit.py runs without crashing.
    """
    result = subprocess.run([sys.executable, "src/submit.py"], capture_output=True)
    assert result.returncode == 0, f"submit.py failed: {result.stderr.decode()}"
