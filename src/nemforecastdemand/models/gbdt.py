"""Gradient-boosted quantile forecaster: the industry point-model foil.

LightGBM regressors on the shared design matrix plus the origin-anchored
recency features, one head per quantile level, trained with the pinball
objective and early stopping on the validation split. Trees handle the
interactions the linear models leave on the table (temperature by hour,
irradiance by season), and the recency columns hand them the same
information the time-series models carry through their AR error dynamics,
which makes this the strongest "just predict the number" benchmark
available for tabular load data.

Honesty notes, also surfaced in notebook 05. The set of quantile heads is
not a generative model: paths cannot be sampled, so the energy score is
unavailable, CRPS is approximated by the quantile integral and the median
head serves as the point forecast. Heads are trained independently, so
quantile crossing is possible and repaired by sorting. Trees cannot
extrapolate beyond the training range of the target, a structural
limitation for trending series.

A separate L2 (squared-error) head is also trained so the model has a
genuine conditional-mean predictor, not just the median quantile. It does
not enter the quantile set or the scoring; it exists only for the
mean-against-median point-forecast comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

from nemforecastdemand.config import Config
from nemforecastdemand.models.base import (
    Forecast,
    Forecaster,
    build_design,
    recency_features,
    stacked_origin_design,
)
from nemforecastdemand.splits import horizon_index

#: Quantile heads. The 0.025/0.975 pair closes the 95% central interval,
#: 0.25/0.75 and 0.1/0.9 close the 50% and 80% intervals, every reporting
#: quantile in the configuration is matched exactly and the even spacing
#: through the body keeps the quantile-integral CRPS accurate.
QUANTILE_LEVELS = (
    0.025,
    0.05,
    0.1,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    0.95,
    0.975,
)

_LGBM_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "verbose": -1,
}


class LightGbmQuantile(Forecaster):
    """One LightGBM head per quantile level over the shared design."""

    name = "lightgbm"

    def __init__(self, cfg: Config, quantiles: tuple[float, ...] = QUANTILE_LEVELS) -> None:
        self.cfg = cfg
        self.quantiles = quantiles
        self._heads: dict[float, LGBMRegressor] = {}
        self._mean_head: LGBMRegressor | None = None
        self.best_iterations: dict[float, int] = {}
        self.mean_best_iteration: int | None = None

    def fit(
        self,
        panel: pd.DataFrame,
        train_origins: pd.DatetimeIndex,
        validation_origins: pd.DatetimeIndex | None = None,
    ) -> LightGbmQuantile:
        """Fit every quantile head, early stopping on a validation window.

        Training rows are origin blocks rather than raw timestamps, so the
        heads see the origin-anchored recency features under exactly the
        distribution they will receive at forecast time.

        Parameters
        ----------
        panel
            Full processed panel.
        train_origins
            Forecast origins whose 48-step blocks form the training rows.
        validation_origins
            Origins whose blocks drive early stopping. When omitted, every
            head runs to the configured estimator cap.
        """
        train_design, train_y = stacked_origin_design(panel, self.cfg, train_origins)
        eval_kwargs = {}
        if validation_origins is not None:
            val_design, val_y = stacked_origin_design(panel, self.cfg, validation_origins)
            eval_kwargs = {
                "eval_set": [(val_design, val_y)],
                "callbacks": [early_stopping(50, verbose=False)],
            }

        for level in self.quantiles:
            head = LGBMRegressor(objective="quantile", alpha=level, **_LGBM_PARAMS)
            head.fit(train_design, train_y, **eval_kwargs)
            self._heads[level] = head
            self.best_iterations[level] = int(head.best_iteration_ or _LGBM_PARAMS["n_estimators"])
            print(
                f"  quantile {level:.3f}: {self.best_iterations[level]} trees",
                flush=True,
            )

        # A genuine conditional-mean head, trained on squared error, so the
        # benchmark has a mean point forecast to set beside its median.
        mean_head = LGBMRegressor(objective="regression", **_LGBM_PARAMS)
        mean_head.fit(train_design, train_y, **eval_kwargs)
        self._mean_head = mean_head
        self.mean_best_iteration = int(mean_head.best_iteration_ or _LGBM_PARAMS["n_estimators"])
        print(f"  mean (L2): {self.mean_best_iteration} trees", flush=True)
        return self

    def forecast(
        self,
        panel: pd.DataFrame,
        origin: pd.Timestamp,
        weather_source: str = "forecast",
        overrides: pd.DataFrame | None = None,
    ) -> Forecast:
        if not self._heads:
            raise RuntimeError("fit the model first")
        index = horizon_index(origin, self.cfg.horizon)
        design = build_design(panel, self.cfg, weather_source=weather_source, overrides=overrides)
        block = pd.concat(
            [design.loc[index], recency_features(panel, origin, self.cfg.horizon)], axis=1
        )
        raw = np.stack([self._heads[level].predict(block) for level in self.quantiles])
        # Independent heads can cross; sorting per step restores a valid
        # quantile function without changing any single head's calibration.
        quantile_values = np.sort(raw, axis=0)
        median = quantile_values[self.quantiles.index(0.5)]
        mean_estimate = None if self._mean_head is None else self._mean_head.predict(block)
        return Forecast(
            origin=origin,
            index=index,
            mean=median,
            quantile_levels=np.asarray(self.quantiles),
            quantile_values=quantile_values,
            mean_estimate=mean_estimate,
        )
