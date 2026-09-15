"""Building and saving the actual-vs-forecast output table."""

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, OUTPUT_PATH


def build_output_frame(y: pd.Series, pred, pred_ci, mape: float) -> pd.DataFrame:
    actual_df = y.reset_index()
    actual_df.columns = ['periode', 'sales']
    actual_df['forecast_sales'] = np.nan
    actual_df['lower_ci'] = np.nan
    actual_df['upper_ci'] = np.nan
    actual_df['mape'] = mape
    actual_df['type'] = 'actual'

    forecast_df = pd.DataFrame({
        'periode': pred.predicted_mean.index,
        'sales': np.nan,
        'forecast_sales': pred.predicted_mean.values,
        'lower_ci': pred_ci.iloc[:, 0].values,
        'upper_ci': pred_ci.iloc[:, 1].values,
        'mape': mape,
        'type': 'forecast'
    })

    return pd.concat([actual_df, forecast_df], ignore_index=True).sort_values('periode')


def save_csv(df: pd.DataFrame, path=OUTPUT_PATH) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
