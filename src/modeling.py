"""SARIMAX model selection and training.

Differencing orders (d, D) are determined upfront via ADF stationarity tests
and held fixed during the grid search, so AIC values stay comparable across
candidates (differencing changes the effective sample size used to compute
the likelihood, which makes AIC incomparable across different d/D).
"""

import itertools

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

SEASONAL_PERIOD = 12


def determine_differencing(y: pd.Series) -> tuple[int, int, dict]:
    adf_level = adfuller(y.dropna())
    d = 0 if adf_level[1] < 0.05 else 1

    y_seasonal_diff = y.diff(SEASONAL_PERIOD).dropna()
    adf_seasonal = adfuller(y_seasonal_diff)
    D = 0 if adf_seasonal[1] < 0.05 else 1

    diagnostics = {
        "adf_level_statistic": adf_level[0],
        "adf_level_pvalue": adf_level[1],
        "adf_seasonal_statistic": adf_seasonal[0],
        "adf_seasonal_pvalue": adf_seasonal[1],
    }
    return d, D, diagnostics


def grid_search(y: pd.Series, d: int, D: int, p_range=range(0, 2), q_range=range(0, 2)) -> pd.DataFrame:
    pdq = [(p, d, q) for p, q in itertools.product(p_range, q_range)]
    seasonal_pdq = [(P, D, Q, SEASONAL_PERIOD) for P, Q in itertools.product(p_range, q_range)]

    results_list = []
    for order in pdq:
        for seasonal_order in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(
                    y,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = mod.fit(disp=False)
                results_list.append({
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "AIC": res.aic
                })
            except Exception:
                continue

    aic_df = pd.DataFrame(results_list)
    return aic_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)


def select_best(aic_df: pd.DataFrame) -> tuple[tuple, tuple]:
    best = aic_df.iloc[0]
    return best["order"], best["seasonal_order"]


def train_sarimax(y: pd.Series, order: tuple, seasonal_order: tuple):
    mod = sm.tsa.statespace.SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return mod.fit(disp=False)
