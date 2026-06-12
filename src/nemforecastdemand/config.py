"""Typed configuration loader over ``config/default.yaml``.

Every tunable in the project lives in the YAML file and is surfaced here as a
frozen dataclass, so downstream code gets attribute access, type checking and
a single source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class GridPoint:
    """Coordinates of the weather grid point."""

    latitude: float
    longitude: float
    label: str


@dataclass(frozen=True)
class Window:
    """Inclusive date window for the study, in market time."""

    start: str
    end: str


@dataclass(frozen=True)
class Splits:
    """Chronological train, validation and test fractions."""

    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class DemandConfig:
    """Demand acquisition settings."""

    source: str


@dataclass(frozen=True)
class WeatherConfig:
    """Weather acquisition and degree-day settings."""

    actuals_model: str
    forecast_model: str
    lead_days: int
    variables: tuple[str, ...]
    heating_base: float
    cooling_base: float


@dataclass(frozen=True)
class FeatureConfig:
    """Design settings: seasonal basis family and size, demand lags."""

    seasonal_basis: str
    daily_harmonics: int
    weekly_harmonics: int
    daily_rbf_centres: int
    weekly_rbf_centres: int
    demand_lags: tuple[int, ...]
    # Daily harmonics interacted with degree days and the weekend flag,
    # letting the temperature response and the weekend profile vary by
    # time of day. Zero disables the block.
    interaction_harmonics: int = 0
    # Basis-function Gaussian process surface over time of day and
    # temperature: a truncated spectral basis whose weight priors carry
    # kernel structure in the model. Zero in either disables the block;
    # only the HSGP variant of the Bayesian model enables it.
    hsgp_time_harmonics: int = 0
    hsgp_temp_basis: int = 0
    hsgp_temp_lo: float = -5.0
    hsgp_temp_hi: float = 45.0


@dataclass(frozen=True)
class ArimaConfig:
    """Baseline settings: the candidate residual orders."""

    candidate_orders: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class BstsPriors:
    """Prior scales for the BSTS, on standardised demand."""

    level_scale: float
    slope_scale: float
    damping_alpha: float
    damping_beta: float
    coef_scale: float
    obs_scale: float
    var_intercept_loc: float
    var_intercept_scale: float
    var_coef_scale: float
    init_level_scale: float
    init_slope_scale: float
    student_t_df_rate: float
    # Beta prior on the AR(1) error persistence in the innovations form.
    ar_alpha: float = 8.0
    ar_beta: float = 2.0


@dataclass(frozen=True)
class BartConfig:
    """Bayesian additive regression trees settings."""

    trees: int
    tune: int
    draws: int
    chains: int


@dataclass(frozen=True)
class BstsConfig:
    """Structural model settings."""

    damped_slope: bool
    obs_family: str
    heteroskedastic: bool
    variance_daily_harmonics: int
    variance_use_degree_days: bool
    priors: BstsPriors


@dataclass(frozen=True)
class ViConfig:
    """ADVI optimisation settings."""

    steps: int
    learning_rate: float
    num_particles: int
    log_every: int
    eval_particles: int
    posterior_draws: int


@dataclass(frozen=True)
class NutsConfig:
    """NUTS sampler settings."""

    chains: int
    warmup: int
    samples: int
    target_accept: float
    max_tree_depth: int
    chain_method: str


@dataclass(frozen=True)
class WarmStartConfig:
    """Cold-versus-warm comparison settings."""

    reduced_warmup: tuple[int, ...]
    target_bulk_ess: float
    rhat_threshold: float


@dataclass(frozen=True)
class PerturbationConfig:
    """Sweep levels for the calibrated covariate perturbation."""

    sweep_multipliers: tuple[float, ...]


@dataclass(frozen=True)
class EvaluationConfig:
    """Scoring settings."""

    quantiles: tuple[float, ...]
    interval_levels: tuple[float, ...]
    crps_chunk: int


@dataclass(frozen=True)
class PathsConfig:
    """Project directories, resolved against the repository root."""

    raw: Path
    interim: Path
    processed: Path
    artifacts: Path
    figures: Path


@dataclass(frozen=True)
class Config:
    """Full project configuration."""

    region: str
    timezone: str
    frequency: str
    horizon: int
    origins: tuple[str, ...]
    grid_point: GridPoint
    window: Window
    splits: Splits
    demand: DemandConfig
    weather: WeatherConfig
    features: FeatureConfig
    arima: ArimaConfig
    bart: BartConfig
    bsts: BstsConfig
    vi: ViConfig
    nuts: NutsConfig
    warm_start: WarmStartConfig
    perturbation: PerturbationConfig
    evaluation: EvaluationConfig
    paths: PathsConfig
    seed: int
    repo_root: Path = field(default=REPO_ROOT)


def _as_int_triples(rows: list[list[int]]) -> tuple[tuple[int, int, int], ...]:
    return tuple((int(p), int(d), int(q)) for p, d, q in rows)


def load_config(path: str | Path | None = None) -> Config:
    """Load the YAML configuration into typed dataclasses.

    Parameters
    ----------
    path
        Location of the YAML file. Defaults to ``config/default.yaml`` at the
        repository root.

    Returns
    -------
    Config
        The fully typed configuration.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(config_path.read_text())
    root = config_path.resolve().parents[1]

    paths = PathsConfig(**{key: root / value for key, value in raw["paths"].items()})
    return Config(
        region=raw["region"],
        timezone=raw["timezone"],
        frequency=raw["frequency"],
        horizon=int(raw["horizon"]),
        origins=tuple(raw["origins"]),
        grid_point=GridPoint(**raw["grid_point"]),
        window=Window(**raw["window"]),
        splits=Splits(**raw["splits"]),
        demand=DemandConfig(**raw["demand"]),
        weather=WeatherConfig(
            actuals_model=raw["weather"]["actuals_model"],
            forecast_model=raw["weather"]["forecast_model"],
            lead_days=int(raw["weather"]["lead_days"]),
            variables=tuple(raw["weather"]["variables"]),
            heating_base=float(raw["weather"]["heating_base"]),
            cooling_base=float(raw["weather"]["cooling_base"]),
        ),
        features=FeatureConfig(
            seasonal_basis=raw["features"]["seasonal_basis"],
            daily_harmonics=int(raw["features"]["daily_harmonics"]),
            weekly_harmonics=int(raw["features"]["weekly_harmonics"]),
            daily_rbf_centres=int(raw["features"]["daily_rbf_centres"]),
            weekly_rbf_centres=int(raw["features"]["weekly_rbf_centres"]),
            demand_lags=tuple(int(lag) for lag in raw["features"]["demand_lags"]),
            interaction_harmonics=int(raw["features"].get("interaction_harmonics", 0)),
        ),
        arima=ArimaConfig(
            candidate_orders=_as_int_triples(raw["arima"]["candidate_orders"]),
        ),
        bart=BartConfig(
            trees=int(raw["bart"]["trees"]),
            tune=int(raw["bart"]["tune"]),
            draws=int(raw["bart"]["draws"]),
            chains=int(raw["bart"]["chains"]),
        ),
        bsts=BstsConfig(
            damped_slope=bool(raw["bsts"]["damped_slope"]),
            obs_family=raw["bsts"]["obs_family"],
            heteroskedastic=bool(raw["bsts"]["heteroskedastic"]),
            variance_daily_harmonics=int(raw["bsts"]["variance_daily_harmonics"]),
            variance_use_degree_days=bool(raw["bsts"]["variance_use_degree_days"]),
            priors=BstsPriors(**raw["bsts"]["priors"]),
        ),
        vi=ViConfig(**raw["vi"]),
        nuts=NutsConfig(**raw["nuts"]),
        warm_start=WarmStartConfig(
            reduced_warmup=tuple(int(w) for w in raw["warm_start"]["reduced_warmup"]),
            target_bulk_ess=float(raw["warm_start"]["target_bulk_ess"]),
            rhat_threshold=float(raw["warm_start"]["rhat_threshold"]),
        ),
        perturbation=PerturbationConfig(
            sweep_multipliers=tuple(float(m) for m in raw["perturbation"]["sweep_multipliers"]),
        ),
        evaluation=EvaluationConfig(
            quantiles=tuple(float(q) for q in raw["evaluation"]["quantiles"]),
            interval_levels=tuple(float(level) for level in raw["evaluation"]["interval_levels"]),
            crps_chunk=int(raw["evaluation"]["crps_chunk"]),
        ),
        paths=paths,
        seed=int(raw["seed"]),
        repo_root=root,
    )
