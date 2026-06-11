"""Seeding, timing and dtype helpers."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import jax
import numpy as np


@dataclass
class TimingRecord:
    """Wall-clock measurement for one labelled step."""

    label: str
    seconds: float = float("nan")


@contextmanager
def timed(label: str, sink: dict[str, float] | None = None) -> Iterator[TimingRecord]:
    """Time a block of code with a monotonic clock.

    Parameters
    ----------
    label
        Name under which the measurement is recorded.
    sink
        Optional mapping that receives ``{label: seconds}`` on exit.

    Yields
    ------
    TimingRecord
        Record whose ``seconds`` field is filled in when the block exits.
    """
    record = TimingRecord(label)
    start = time.perf_counter()
    try:
        yield record
    finally:
        record.seconds = time.perf_counter() - start
        if sink is not None:
            sink[label] = record.seconds


@dataclass
class JaxTiming:
    """First-call and steady-state timings for a JIT-compiled function.

    The first call includes XLA compilation, so the compile cost is reported
    separately rather than inflating the run time.
    """

    first_seconds: float
    run_seconds: float
    compile_seconds: float = field(init=False)

    def __post_init__(self) -> None:
        self.compile_seconds = max(self.first_seconds - self.run_seconds, 0.0)


def time_jax(fn: Callable, *args, **kwargs) -> tuple[object, JaxTiming]:
    """Time a JAX callable, separating compilation from execution.

    The function is called twice with identical arguments. The first call
    pays compilation, the second measures steady-state execution. Both calls
    block until results are ready, so device work is fully accounted for.

    Parameters
    ----------
    fn
        The callable to time. It must be deterministic in its timing-relevant
        behaviour (same shapes and dtypes on both calls).
    *args, **kwargs
        Arguments forwarded to ``fn``.

    Returns
    -------
    tuple
        The result of the second call and the timing record.
    """
    start = time.perf_counter()
    jax.block_until_ready(fn(*args, **kwargs))
    first = time.perf_counter() - start

    start = time.perf_counter()
    result = jax.block_until_ready(fn(*args, **kwargs))
    run = time.perf_counter() - start
    return result, JaxTiming(first_seconds=first, run_seconds=run)


def rng_key(seed: int) -> jax.Array:
    """Create a JAX PRNG key from an integer seed."""
    return jax.random.PRNGKey(seed)


def numpy_rng(seed: int) -> np.random.Generator:
    """Create a NumPy generator from an integer seed."""
    return np.random.default_rng(seed)


def tree_to_float32(tree: object) -> object:
    """Cast every floating array in a pytree to float32.

    Posterior draws are stored and post-processed in float32. Halving the
    footprint matters once draws are stacked across chains and time.
    """

    def cast(leaf: object) -> object:
        if isinstance(leaf, np.ndarray | jax.Array) and np.issubdtype(leaf.dtype, np.floating):
            return leaf.astype(np.float32)
        return leaf

    return jax.tree.map(cast, tree)
