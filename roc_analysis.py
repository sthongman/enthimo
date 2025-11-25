"""Command-line ROC analysis for grouped CSV datasets.

Features:
- Loads a CSV file containing a label column (e.g., groups 0/1/2) and numeric predictors.
- Computes sensitivity, specificity (Youden's J optimal threshold), AUC, and p-value
  (DeLong et al., 1988 nonparametric test versus AUC=0.5) for each numeric feature.
- Handles all pairwise class comparisons by default or a user-specified class pair.
- Saves results to stdout or an optional CSV file.

Example
-------
python roc_analysis.py data.csv --label group
python roc_analysis.py data.csv --label group --classes 1 0 --features biomarker1 biomarker2
python roc_analysis.py data.csv --label group --output results.csv
"""
from __future__ import annotations

import argparse
import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Computes midranks needed for DeLong AUC variance.

    Parameters
    ----------
    x: np.ndarray
        1D array of scores.
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fast implementation of DeLong's method for ROC AUC.

    Parameters
    ----------
    predictions_sorted_transposed : np.ndarray
        Sorted scores (descending) for positives first, shape (n_classifiers, n_samples)
    label_1_count : int
        Number of positive samples.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_predictions = predictions_sorted_transposed[:, :m]
    negative_predictions = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.array([_compute_midrank(x) for x in positive_predictions])
    ty = np.array([_compute_midrank(x) for x in negative_predictions])
    tz = np.array([_compute_midrank(x) for x in predictions_sorted_transposed])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_variance(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Calculates DeLong et al. (1988) AUC and variance for binary classification."""
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]
    label_1_count = int(np.sum(y_true_sorted))
    predictions_sorted_transposed = y_score_sorted[np.newaxis, :]
    aucs, delongcov = _fast_delong(predictions_sorted_transposed, label_1_count)
    return aucs[0], delongcov[0, 0]


def delong_p_value(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Two-sided p-value for AUC different from 0.5 using the DeLong et al. test."""
    auc, auc_var = delong_roc_variance(y_true, y_score)
    if auc_var <= 0:
        return float("nan")
    z = abs(auc - 0.5) / np.sqrt(auc_var)
    return 2 * stats.norm.sf(z)


def youden_index(tpr: np.ndarray, fpr: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float, float]:
    """Returns (sensitivity, specificity, threshold) maximizing Youden's J."""
    j_scores = tpr - fpr
    best_idx = int(np.nanargmax(j_scores))
    return float(tpr[best_idx]), float(1 - fpr[best_idx]), float(thresholds[best_idx])


def analyze_feature(
    feature: str,
    values: pd.Series,
    labels: pd.Series,
    positive_label,
    negative_label,
) -> dict:
    """Compute ROC metrics for a single feature."""
    clean = pd.concat([values, labels], axis=1).dropna()
    if clean.empty:
        raise ValueError(f"Feature {feature} has no valid rows after dropping NaNs")
    y_true = (clean.iloc[:, 1] == positive_label).astype(int).to_numpy()
    scores = clean.iloc[:, 0].astype(float).to_numpy()

    auc = roc_auc_score(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    sensitivity, specificity, threshold = youden_index(tpr, fpr, thresholds)
    p_value = delong_p_value(y_true, scores)

    return {
        "feature": feature,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": threshold,
        "p_value": p_value,
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
    }


def iter_class_pairs(unique_labels: Sequence, forced_pair: Sequence | None = None) -> Iterable[Tuple]:
    if forced_pair:
        if len(forced_pair) != 2:
            raise ValueError("--classes expects exactly two labels")
        yield forced_pair[0], forced_pair[1]
    else:
        yield from itertools.combinations(unique_labels, 2)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch ROC analysis for CSV datasets.")
    parser.add_argument("csv", type=str, help="Path to input CSV file")
    parser.add_argument("--label", required=True, help="Name of the label column")
    parser.add_argument(
        "--features",
        nargs="*",
        help="Optional list of feature columns to analyze (defaults to all numeric columns except label)",
    )
    parser.add_argument("--classes", nargs=2, help="Pair of class labels (positive negative) to compare")
    parser.add_argument("--output", help="Optional path to save results as CSV")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    df = pd.read_csv(args.csv)

    if args.label not in df.columns:
        raise ValueError(f"Label column '{args.label}' not found in CSV")

    feature_cols = args.features
    if not feature_cols:
        feature_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != args.label]
        if not feature_cols:
            raise ValueError("No numeric feature columns found to analyze")

    results = []
    unique_labels = df[args.label].dropna().unique().tolist()

    for pos_label, neg_label in iter_class_pairs(unique_labels, args.classes):
        subset = df[df[args.label].isin([pos_label, neg_label])]
        label_series = subset[args.label]
        for feature in feature_cols:
            metrics = analyze_feature(feature, subset[feature], label_series, pos_label, neg_label)
            results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df[
        [
            "feature",
            "positive_label",
            "negative_label",
            "auc",
            "sensitivity",
            "specificity",
            "threshold",
            "p_value",
            "n_positive",
            "n_negative",
        ]
    ]

    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
    print(results_df.to_string(index=False))

    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\nSaved results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
