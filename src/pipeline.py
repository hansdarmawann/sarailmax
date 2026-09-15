"""Leakage-free training, validation, and future forecasting pipeline."""

import argparse
from pathlib import Path

from src import data, evaluation, modeling
from src import export, setup
from src.config import FORECAST_MONTHS, OUTPUT_PATH, VALIDATION_MONTHS


def run(y, validation_months=VALIDATION_MONTHS, forecast_months=FORECAST_MONTHS):
    """Evaluate on a time-ordered holdout, then retrain and forecast.

    Model selection and parameter estimation use only the training portion.
    The validation portion is used once for an honest out-of-sample score.
    """
    if len(y) <= validation_months:
        raise ValueError("The series must be longer than the validation horizon.")

    train = y.iloc[:-validation_months].copy()
    validation = y.iloc[-validation_months:].copy()

    d, D, diagnostics = modeling.determine_differencing(train)
    aic_df = modeling.grid_search(train, d, D)
    order, seasonal_order = modeling.select_best(aic_df)

    validation_model = modeling.train_sarimax(train, order, seasonal_order)
    validation_prediction = validation_model.get_forecast(steps=validation_months)
    mape = evaluation.compute_mape(validation, validation_prediction)

    final_model = modeling.train_sarimax(y, order, seasonal_order)
    future_prediction, future_ci = evaluation.forecast_future(
        final_model, forecast_months
    )

    return {
        "train": train,
        "validation": validation,
        "order": order,
        "seasonal_order": seasonal_order,
        "diagnostics": diagnostics,
        "aic": aic_df,
        "validation_prediction": validation_prediction,
        "mape": mape,
        "results": final_model,
        "prediction": future_prediction,
        "prediction_ci": future_ci,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Superstore SARIMAX forecast pipeline.")
    parser.add_argument("--validation-months", type=int, default=VALIDATION_MONTHS)
    parser.add_argument("--forecast-months", type=int, default=FORECAST_MONTHS)
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="CSV output path")
    args = parser.parse_args()

    if args.validation_months < 1 or args.forecast_months < 1:
        parser.error("month arguments must be positive integers")

    setup.ensure_directories()
    if not setup.verify_data_file():
        data.download_demo_dataset()

    raw = data.load_raw_data()
    cleaned = data.preprocess(raw)
    y, _ = data.aggregate_monthly(cleaned)
    result = run(y, args.validation_months, args.forecast_months)

    output = export.build_output_frame(
        y,
        result["prediction"],
        result["prediction_ci"],
        result["mape"],
    )
    export.save_csv(output, Path(args.output))
    print(f"MAPE: {result['mape']:.2f}%")
    print(f"Model: order={result['order']}, seasonal_order={result['seasonal_order']}")


if __name__ == "__main__":
    main()
