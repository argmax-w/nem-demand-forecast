"""Weather features and the calibrated forecast perturbation.

The perturbation generator drives the robustness sweep in notebook 05. It
adds a temporally correlated AR(1) error to actual weather, with a
step-dependent standard deviation fitted to the measured forecast-minus-ERA5
residuals, so "degraded input" means degraded in the way real forecasts
degrade. The schedule is indexed by half hour of the market day, which over a
24-hour horizon is a close proxy for lead time and captures the diurnal cycle
of predictability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nemforecastdemand.data.loaders import MARKET_TZ


def degree_days(temp: pd.Series, heating_base: float, cooling_base: float) -> pd.DataFrame:
    """Half-hourly heating and cooling degrees.

    Parameters
    ----------
    temp
        Temperature in degrees Celsius.
    heating_base, cooling_base
        Comfort band edges; demand rises as temperature leaves the band.

    Returns
    -------
    pandas.DataFrame
        Columns ``cooling_deg`` and ``heating_deg``, zero inside the band.
    """
    values = temp.to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {
            "cooling_deg": np.maximum(values - cooling_base, 0.0),
            "heating_deg": np.maximum(heating_base - values, 0.0),
        },
        index=temp.index,
    )


@dataclass(frozen=True)
class PerturbationModel:
    """Fitted AR(1) error model for one weather variable.

    Attributes
    ----------
    rho
        Lag-1 autocorrelation of forecast residuals on the half-hourly grid.
    sigma_by_step
        Residual standard deviation for each half hour of the market day.
    nonnegative
        Whether perturbed values are clipped at zero (irradiance).
    """

    rho: float
    sigma_by_step: np.ndarray
    nonnegative: bool = False

    def sample(
        self,
        actual: np.ndarray,
        steps: np.ndarray,
        multiplier: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Perturb a path of actuals with a correlated error draw.

        Parameters
        ----------
        actual
            Actual values over the forecast horizon.
        steps
            Half-hour-of-market-day index for each horizon position.
        multiplier
            Error magnitude in multiples of the fitted schedule. Zero
            returns the actuals unchanged (perfect foresight).
        rng
            Source of randomness, seeded by the caller per origin.

        Returns
        -------
        numpy.ndarray
            The perturbed path.
        """
        sigma = self.sigma_by_step[steps] * multiplier
        shocks = rng.standard_normal(len(actual))
        errors = np.empty(len(actual))
        scale = np.sqrt(1.0 - self.rho**2)
        errors[0] = sigma[0] * shocks[0]
        for i in range(1, len(actual)):
            errors[i] = self.rho * errors[i - 1] + sigma[i] * scale * shocks[i]
        out = actual + errors
        if self.nonnegative:
            out = np.maximum(out, 0.0)
        return out


def fit_perturbation(
    actual: pd.Series,
    forecast: pd.Series,
    nonnegative: bool = False,
    smooth_window: int = 5,
) -> PerturbationModel:
    """Fit the AR(1) error model from measured forecast residuals.

    Parameters
    ----------
    actual, forecast
        Aligned half-hourly series on the UTC grid (training window only, so
        the sweep is calibrated without touching test data).
    nonnegative
        Clip perturbed values at zero when sampling.
    smooth_window
        Rolling window (in half hours) applied to the per-step standard
        deviations to steady the small per-bucket samples.

    Returns
    -------
    PerturbationModel
        The fitted error model.
    """
    aligned = pd.DataFrame({"actual": actual, "forecast": forecast}).dropna()
    residual = aligned["forecast"] - aligned["actual"]
    rho = float(residual.autocorr(lag=1))

    market_index = residual.index.tz_convert(MARKET_TZ)
    step = market_index.hour * 2 + market_index.minute // 30
    sigma = residual.groupby(step).std().reindex(range(48)).ffill().bfill()
    sigma = (
        sigma.rolling(smooth_window, center=True, min_periods=1).mean().to_numpy(dtype=np.float64)
    )
    return PerturbationModel(rho=rho, sigma_by_step=sigma, nonnegative=nonnegative)
