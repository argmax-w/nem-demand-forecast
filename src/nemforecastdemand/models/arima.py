"""Dynamic harmonic regression with ARIMA errors via statsmodels SARIMAX.

The classical baseline. Seasonality, weather, lagged demand and holidays are
exogenous regressors; a low-order stationary ARMA process carries what they
miss. A bare ARIMA on raw half-hourly demand is deliberately avoided as a
strawman: with two seasonalities and strong weather dependence it would not
be a serious operational candidate.

The residual order is selected on the validation set (notebook 02). The
fitted Gaussian state-space model yields an analytic Gaussian predictive,
scored with the closed-form CRPS, which doubles as the correctness reference
for the sample-based estimator used by the Bayesian models. Because that
predictive is a joint Gaussian over the whole horizon, the model can also
simulate coherent sample paths, so the energy score over whole days is
defined for the classical baseline too.

Forecasting from an origin re-applies the fitted parameters to all demand
history up to that origin (a Kalman filter pass, no refit), with realised
weather behind the origin and the chosen weather variant ahead of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from nemforecastdemand.config import Config
from nemforecastdemand.models.base import Forecast, Forecaster, build_design
from nemforecastdemand.splits import horizon_index


class DynamicHarmonicRegression(Forecaster):
    """SARIMAX with the shared design matrix as exogenous regressors."""

    name = "arima"

    def __init__(self, cfg: Config, order: tuple[int, int, int]) -> None:
        self.cfg = cfg
        self.order = order
        self._results = None
        self._fit_start: pd.Timestamp | None = None
        self._y_loc = 0.0
        self._y_scale = 1.0
        self._x_loc: pd.Series | None = None
        self._x_scale: pd.Series | None = None

    def _standardise_design(self, design: pd.DataFrame) -> pd.DataFrame:
        return (design - self._x_loc) / self._x_scale

    def fit(self, panel: pd.DataFrame, fit_index: pd.DatetimeIndex) -> DynamicHarmonicRegression:
        """Fit by maximum likelihood on the given window.

        Demand and regressors are standardised on the window for numerical
        conditioning; the scales are stored and undone at forecast time.
        """
        design = build_design(panel, self.cfg, weather_source="actual").loc[fit_index]
        if design.isna().any().any():
            raise ValueError("fit window starts inside the demand-lag warmup")
        y = panel["demand_mw"].loc[fit_index].astype(np.float64)

        self._y_loc, self._y_scale = float(y.mean()), float(y.std())
        self._x_loc = design.mean()
        self._x_scale = design.std().replace(0.0, 1.0)
        self._fit_start = fit_index[0]

        endog = (y - self._y_loc) / self._y_scale
        endog.index.freq = "30min"
        model = SARIMAX(
            endog,
            exog=self._standardise_design(design),
            order=self.order,
            trend="c",
            concentrate_scale=True,
        )
        # lbfgs flags non-convergence at its strict gradient tolerance even
        # once the likelihood has plateaued; the fit script verifies the
        # plateau rather than chasing the tolerance.
        self._results = model.fit(disp=False, method="lbfgs", maxiter=500)
        return self

    @property
    def results(self):
        """The statsmodels results object, for inspection in notebooks."""
        if self._results is None:
            raise RuntimeError("fit the model first")
        return self._results

    def forecast(
        self,
        panel: pd.DataFrame,
        origin: pd.Timestamp,
        weather_source: str = "forecast",
        overrides: pd.DataFrame | None = None,
    ) -> Forecast:
        if self._results is None:
            raise RuntimeError("fit the model first")
        index = horizon_index(origin, self.cfg.horizon)

        history = panel.index[(panel.index >= self._fit_start) & (panel.index < origin)]
        design_history = build_design(panel, self.cfg, weather_source="actual").loc[history]
        design_future = build_design(
            panel, self.cfg, weather_source=weather_source, overrides=overrides
        ).loc[index]

        y_history = (
            panel["demand_mw"].loc[history].astype(np.float64) - self._y_loc
        ) / self._y_scale
        y_history.index.freq = "30min"
        applied = self._results.apply(
            y_history, exog=self._standardise_design(design_history), refit=False
        )
        prediction = applied.get_forecast(
            steps=self.cfg.horizon, exog=self._standardise_design(design_future)
        )
        mean = prediction.predicted_mean.to_numpy() * self._y_scale + self._y_loc
        sd = np.sqrt(prediction.var_pred_mean.to_numpy()) * self._y_scale
        return Forecast(origin=origin, index=index, mean=mean, sd=sd)

    def simulate_paths(
        self,
        panel: pd.DataFrame,
        origin: pd.Timestamp,
        weather_source: str = "forecast",
        n_paths: int = 1000,
        seed: int = 0,
        overrides: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Coherent predictive sample paths over the horizon, ``(n_paths, H)`` in MW.

        The fitted state-space model is a joint Gaussian over the 48 steps, not a
        stack of per-step marginals. Simulating forward from the filtered
        end-of-sample state with fresh innovations therefore draws whole-day
        trajectories whose cross-step dependence is the AR error's own. The
        marginal mean and variance of these paths match the analytic
        ``get_forecast`` predictive; what they add is the joint law a sample-only
        score such as the energy score needs.
        """
        if self._results is None:
            raise RuntimeError("fit the model first")
        index = horizon_index(origin, self.cfg.horizon)

        history = panel.index[(panel.index >= self._fit_start) & (panel.index < origin)]
        design_history = build_design(panel, self.cfg, weather_source="actual").loc[history]
        design_future = build_design(
            panel, self.cfg, weather_source=weather_source, overrides=overrides
        ).loc[index]

        y_history = (
            panel["demand_mw"].loc[history].astype(np.float64) - self._y_loc
        ) / self._y_scale
        y_history.index.freq = "30min"
        applied = self._results.apply(
            y_history, exog=self._standardise_design(design_history), refit=False
        )
        sim = applied.simulate(
            nsimulations=self.cfg.horizon,
            anchor="end",
            exog=self._standardise_design(design_future),
            repetitions=n_paths,
            random_state=seed,
        )
        paths = np.asarray(sim, dtype=np.float64).reshape(self.cfg.horizon, n_paths).T
        return paths * self._y_scale + self._y_loc
