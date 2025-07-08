import importlib

# Import train.py as a module
train = importlib.import_module("src.train")


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
