"""Configuration loader and utility checks."""

import numpy as np

from nemforecastdemand.config import load_config
from nemforecastdemand.utils import numpy_rng, timed, tree_to_float32


def test_default_config_loads_and_is_typed():
    cfg = load_config()
    assert cfg.region == "NSW1"
    assert cfg.horizon == 48
    assert isinstance(cfg.arima.candidate_orders[0], tuple)
    assert all(len(order) == 3 for order in cfg.arima.candidate_orders)
    assert cfg.paths.processed.name == "processed"
    assert cfg.paths.processed.is_absolute()


def test_split_fractions_sum_to_one():
    cfg = load_config()
    total = cfg.splits.train + cfg.splits.validation + cfg.splits.test
    assert abs(total - 1.0) < 1e-9


def test_timed_records_elapsed():
    sink: dict[str, float] = {}
    with timed("step", sink) as record:
        sum(range(1000))
    assert record.seconds >= 0.0
    assert sink["step"] == record.seconds


def test_numpy_rng_is_deterministic():
    a = numpy_rng(7).normal(size=4)
    b = numpy_rng(7).normal(size=4)
    np.testing.assert_array_equal(a, b)


def test_tree_to_float32_casts_floats_only():
    tree = {"draws": np.ones(3, dtype=np.float64), "count": np.arange(3)}
    out = tree_to_float32(tree)
    assert out["draws"].dtype == np.float32
    assert out["count"].dtype == np.arange(3).dtype
