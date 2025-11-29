"""Multiple linear regression for HFC with five predictors.

This script loads a CSV file that contains HFC (dependent variable) and five
independent variables (age, MMSE, PCS, DCS, R2star). DCS is converted to its
absolute value before fitting the model. The script fits an ordinary least
squares regression and prints a concise coefficient table. Optionally, the
results can be saved as a CSV file.
"""
from __future__ import annotations

import argparse
from typing import Iterable, List

import pandas as pd
import statsmodels.api as sm


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multiple linear regression of HFC on age, MMSE, PCS, |DCS|, and R2star."
        )
    )
    parser.add_argument("csv", help="Path to the CSV file containing the data")
    parser.add_argument(
        "--target",
        default="HFC",
        help="Column name for the dependent variable (default: HFC)",
    )
    parser.add_argument(
        "--predictors",
        nargs=5,
        metavar=("age", "MMSE", "PCS", "DCS", "R2star"),
        default=["age", "MMSE", "PCS", "DCS", "R2star"],
        help="Five predictor column names in order (default: age MMSE PCS DCS R2star)",
    )
    parser.add_argument(
        "--output",
        help="Optional CSV path to save the coefficient table",
    )
    return parser.parse_args(argv)


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {', '.join(missing)}")


def fit_regression(df: pd.DataFrame, target: str, predictors: List[str]) -> pd.DataFrame:
    validate_columns(df, [target, *predictors])

    data = df[[target, *predictors]].dropna().copy()
    if not len(data):
        raise ValueError("No rows remain after dropping NaNs")

    if "DCS" in data.columns:
        data["DCS"] = data["DCS"].abs()

    y = data[target].astype(float)
    X = data[predictors].astype(float)
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    summary_df = model.summary2().tables[1].reset_index().rename(columns={"index": "term"})
    summary_df = summary_df.rename(
        columns={
            "Coef.": "coef",
            "Std.Err.": "std_err",
            "[0.025": "ci_lower",
            "0.975]": "ci_upper",
        }
    )
    summary_df = summary_df[
        ["term", "coef", "std_err", "t", "P>|t|", "ci_lower", "ci_upper"]
    ]
    return summary_df


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    df = pd.read_csv(args.csv)
    summary_df = fit_regression(df, args.target, args.predictors)

    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
    print(summary_df.to_string(index=False))

    if args.output:
        summary_df.to_csv(args.output, index=False)
        print(f"\nSaved coefficients to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
