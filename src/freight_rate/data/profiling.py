#!/usr/bin/env python3

"""
Read-only profiler for the Spotter freight-rate ML assessment.

Usage:
    python profiling.py

Expected files:
    data/train_test.csv
    data/validation.csv
    data/validation_predictions_template.csv
    data/december_chart_inputs.csv

Output:
    dataset_profile.txt

The script does NOT:
    - modify input CSVs
    - train models
    - install packages
    - make network requests
    - upload data

By default, it does NOT write actual dataset rows to the report.
"""

from __future__ import annotations

import math
import platform
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
OUTPUT_FILE = Path("dataset_profile.txt")

FILES = {
    "train": DATA_DIR / "train_test.csv",
    "validation": DATA_DIR / "validation.csv",
    "prediction_template": DATA_DIR / "validation_predictions_template.csv",
    "december": DATA_DIR / "december_chart_inputs.csv",
}

# Set this to True only if you are comfortable sharing 10 anonymized rows.
INCLUDE_SAMPLE_ROWS = False
SAMPLE_ROWS = 10

# Maximum number of categorical values shown for any column.
TOP_CATEGORIES = 15

# Columns whose names strongly suggest they may be IDs.
ID_NAME_HINTS = (
    "id",
    "load_id",
    "shipment_id",
    "order_id",
    "record_id",
)

# Columns whose names strongly suggest they may contain dates.
DATE_NAME_HINTS = (
    "date",
    "datetime",
    "timestamp",
    "time",
)


def fmt(value) -> str:
    """Human-readable formatting."""
    if pd.isna(value):
        return "NaN"

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return str(value)
        return f"{value:.6g}"

    return str(value)


def safe_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def is_numeric_dtype(dtype) -> bool:
    return pd.api.types.is_numeric_dtype(dtype)


def looks_like_date_column(name: str, series: pd.Series) -> bool:
    name_lower = name.lower()

    if any(hint in name_lower for hint in DATE_NAME_HINTS):
        return True

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if pd.api.types.is_object_dtype(series):
        # Only inspect a limited sample for date-like strings.
        sample = series.dropna().astype(str).head(1000)

        if len(sample) == 0:
            return False

        parsed = pd.to_datetime(sample, errors="coerce")

        return parsed.notna().mean() >= 0.90

    return False


def looks_like_id_column(name: str) -> bool:
    name_lower = name.lower()

    if name_lower in ID_NAME_HINTS:
        return True

    return (
        name_lower.endswith("_id")
        or name_lower.endswith("id")
        or "identifier" in name_lower
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path, low_memory=False)


def describe_file(label: str, path: Path, df: pd.DataFrame, out: list[str]) -> None:
    out.append("=" * 90)
    out.append(f"FILE: {label}")
    out.append("=" * 90)

    out.append(f"Path: {path}")
    out.append(f"File size (MB): {path.stat().st_size / (1024 ** 2):.3f}")
    out.append(f"Rows: {len(df):,}")
    out.append(f"Columns: {len(df.columns):,}")
    out.append("")

    out.append("COLUMNS:")
    for i, column in enumerate(df.columns, start=1):
        out.append(
            f"  {i:>3}. {column} | "
            f"dtype={df[column].dtype} | "
            f"unique={df[column].nunique(dropna=True):,} | "
            f"missing={df[column].isna().sum():,} "
            f"({safe_pct(df[column].isna().sum(), len(df)):.3f}%)"
        )

    out.append("")

    duplicate_rows = int(df.duplicated().sum())

    out.append("DUPLICATES:")
    out.append(f"  Duplicate complete rows: {duplicate_rows:,}")

    potential_ids = [c for c in df.columns if looks_like_id_column(c)]

    if potential_ids:
        for column in potential_ids:
            duplicated = int(df[column].duplicated(keep=False).sum())
            missing = int(df[column].isna().sum())

            out.append(
                f"  Potential ID column '{column}': "
                f"missing={missing:,}, "
                f"duplicated values={duplicated:,}"
            )
    else:
        out.append("  Potential ID columns: none detected by name")

    out.append("")

    out.append("NUMERIC COLUMNS:")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_columns:
        out.append("  None")
    else:
        for column in numeric_columns:
            series = pd.to_numeric(df[column], errors="coerce")

            finite = series[np.isfinite(series)]

            if len(finite) == 0:
                out.append(f"  {column}: no finite numeric values")
                continue

            out.append(
                f"  {column}: "
                f"count={len(finite):,}, "
                f"mean={fmt(finite.mean())}, "
                f"std={fmt(finite.std())}, "
                f"min={fmt(finite.min())}, "
                f"p01={fmt(finite.quantile(0.01))}, "
                f"p25={fmt(finite.quantile(0.25))}, "
                f"median={fmt(finite.median())}, "
                f"p75={fmt(finite.quantile(0.75))}, "
                f"p99={fmt(finite.quantile(0.99))}, "
                f"max={fmt(finite.max())}"
            )

            non_finite = int((~np.isfinite(series.fillna(0))).sum())
            if non_finite:
                out.append(
                    f"      WARNING: non-finite numeric values={non_finite:,}"
                )

            negative = int((finite < 0).sum())
            zero = int((finite == 0).sum())

            if negative or zero:
                out.append(
                    f"      Negative values={negative:,}; "
                    f"zero values={zero:,}"
                )

    out.append("")

    out.append("CATEGORICAL / OBJECT COLUMNS:")

    categorical_columns = [
        c
        for c in df.columns
        if not is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]

    if not categorical_columns:
        out.append("  None")
    else:
        for column in categorical_columns:
            series = df[column]

            out.append(
                f"  {column}: "
                f"unique={series.nunique(dropna=True):,}, "
                f"missing={series.isna().sum():,}"
            )

            top = series.value_counts(dropna=False).head(TOP_CATEGORIES)

            for value, count in top.items():
                value_repr = "<NA>" if pd.isna(value) else repr(str(value))
                out.append(
                    f"      {value_repr}: {count:,} "
                    f"({safe_pct(count, len(df)):.2f}%)"
                )

    out.append("")

    out.append("DATE-LIKE COLUMNS:")

    date_columns = [
        c for c in df.columns
        if looks_like_date_column(c, df[c])
    ]

    if not date_columns:
        out.append("  None detected")
    else:
        for column in date_columns:
            parsed = pd.to_datetime(df[column], errors="coerce")
            valid = parsed.dropna()

            if len(valid) == 0:
                out.append(f"  {column}: could not parse dates")
                continue

            out.append(
                f"  {column}: "
                f"valid={len(valid):,}, "
                f"invalid={parsed.isna().sum():,}, "
                f"min={valid.min()}, "
                f"max={valid.max()}, "
                f"unique_dates={valid.nunique():,}"
            )

            counts = valid.dt.to_period("M").value_counts().sort_index()

            out.append("      Rows by month:")
            for month, count in counts.items():
                out.append(f"        {month}: {count:,}")

    out.append("")

    if INCLUDE_SAMPLE_ROWS:
        out.append("SAMPLE ROWS:")
        out.append(
            df.head(SAMPLE_ROWS)
            .to_string(index=False)
        )
        out.append("")

    # Memory cleanup is not strictly necessary, but helps for large files.
    del df


def correlation_analysis(
    train: pd.DataFrame,
    out: list[str],
) -> None:
    out.append("=" * 90)
    out.append("NUMERIC CORRELATION ANALYSIS")
    out.append("=" * 90)

    numeric = train.select_dtypes(include=[np.number])

    if numeric.shape[1] < 2:
        out.append("Not enough numeric columns for correlation analysis.")
        out.append("")
        return

    corr = numeric.corr(numeric_only=True)

    out.append("Correlation matrix:")
    out.append(corr.round(3).to_string())

    out.append("")

    # Try to identify a likely target.
    target_candidates = [
        c for c in train.columns
        if any(
            word in c.lower()
            for word in (
                "rate",
                "price",
                "cost",
                "target",
                "freight",
                "amount",
                "revenue",
            )
        )
    ]

    out.append(f"Potential target candidates: {target_candidates}")

    for target in target_candidates:
        if target not in corr.columns:
            continue

        correlations = (
            corr[target]
            .drop(labels=[target], errors="ignore")
            .abs()
            .sort_values(ascending=False)
        )

        out.append("")
        out.append(
            f"Absolute numeric correlations with candidate target '{target}':"
        )

        for feature, value in correlations.items():
            out.append(f"  {feature}: {value:.4f}")

    out.append("")


def overlap_analysis(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    out: list[str],
) -> None:
    out.append("=" * 90)
    out.append("TRAIN / VALIDATION OVERLAP ANALYSIS")
    out.append("=" * 90)

    common_columns = [
        c for c in train.columns
        if c in validation.columns
    ]

    out.append(f"Common columns: {len(common_columns)}")

    # Candidate categorical columns.
    categorical = [
        c for c in common_columns
        if (
            train[c].dtype == "object"
            or validation[c].dtype == "object"
            or pd.api.types.is_categorical_dtype(train[c])
            or pd.api.types.is_categorical_dtype(validation[c])
        )
    ]

    for column in categorical:
        train_values = set(train[column].dropna().astype(str))
        valid_values = set(validation[column].dropna().astype(str))

        intersection = train_values & valid_values
        validation_only = valid_values - train_values
        train_only = train_values - valid_values

        out.append("")
        out.append(f"Column: {column}")
        out.append(f"  Train unique: {len(train_values):,}")
        out.append(f"  Validation unique: {len(valid_values):,}")
        out.append(f"  Shared: {len(intersection):,}")
        out.append(f"  Validation-only: {len(validation_only):,}")
        out.append(f"  Train-only: {len(train_only):,}")

        if validation_only:
            examples = sorted(validation_only)[:20]
            out.append(
                f"  Validation-only examples: {examples}"
            )

    # Route overlap if pickup/delivery are available.
    pickup_candidates = [
        c for c in common_columns
        if c.lower() in {"pickup", "origin", "origin_city", "pickup_city"}
    ]

    delivery_candidates = [
        c for c in common_columns
        if c.lower() in {"delivery", "destination", "destination_city", "delivery_city"}
    ]

    if pickup_candidates and delivery_candidates:
        pickup = pickup_candidates[0]
        delivery = delivery_candidates[0]

        train_routes = (
            train[pickup].astype(str)
            + " -> "
            + train[delivery].astype(str)
        )

        valid_routes = (
            validation[pickup].astype(str)
            + " -> "
            + validation[delivery].astype(str)
        )

        train_route_set = set(train_routes)
        valid_route_set = set(valid_routes)

        shared = train_route_set & valid_route_set

        out.append("")
        out.append("ROUTE OVERLAP:")
        out.append(f"  Pickup column: {pickup}")
        out.append(f"  Delivery column: {delivery}")
        out.append(f"  Unique train routes: {len(train_route_set):,}")
        out.append(f"  Unique validation routes: {len(valid_route_set):,}")
        out.append(f"  Shared routes: {len(shared):,}")
        out.append(
            f"  Validation routes unseen during training: "
            f"{len(valid_route_set - train_route_set):,}"
        )
    else:
        out.append("")
        out.append(
            "ROUTE OVERLAP: pickup/delivery columns not confidently detected."
        )

    out.append("")


def target_analysis(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    out: list[str],
) -> None:
    out.append("=" * 90)
    out.append("TARGET CANDIDATE ANALYSIS")
    out.append("=" * 90)

    candidates = []

    for column in train.columns:
        name = column.lower()

        score = 0

        if any(x in name for x in ["rate", "price", "cost", "target"]):
            score += 3

        if any(x in name for x in ["freight", "amount"]):
            score += 2

        if column not in validation.columns:
            score += 4

        if is_numeric_dtype(train[column]):
            score += 1

        if score > 0:
            candidates.append((score, column))

    candidates.sort(reverse=True)

    if not candidates:
        out.append("No obvious target candidate detected.")
        out.append("")
        return

    out.append("Potential target candidates:")
    for score, column in candidates:
        train_series = train[column]

        out.append(
            f"  {column}: heuristic_score={score}, "
            f"dtype={train_series.dtype}, "
            f"missing={train_series.isna().sum():,}, "
            f"unique={train_series.nunique(dropna=True):,}"
        )

        if is_numeric_dtype(train_series):
            values = pd.to_numeric(train_series, errors="coerce").dropna()

            if len(values):
                out.append(
                    f"      mean={fmt(values.mean())}, "
                    f"std={fmt(values.std())}, "
                    f"min={fmt(values.min())}, "
                    f"median={fmt(values.median())}, "
                    f"max={fmt(values.max())}"
                )

                for percentile in [0.01, 0.05, 0.95, 0.99]:
                    out.append(
                        f"      p{int(percentile * 100):02d}="
                        f"{fmt(values.quantile(percentile))}"
                    )

                out.append(
                    f"      <=0 values={(values <= 0).sum():,}"
                )

    out.append("")


def december_analysis(
    december: pd.DataFrame,
    out: list[str],
) -> None:
    out.append("=" * 90)
    out.append("DECEMBER INPUT ANALYSIS")
    out.append("=" * 90)

    out.append(f"Rows: {len(december):,}")
    out.append(f"Columns: {len(december.columns):,}")
    out.append(f"Columns: {list(december.columns)}")

    for column in december.columns:
        series = december[column]

        out.append(
            f"  {column}: "
            f"unique={series.nunique(dropna=True):,}, "
            f"missing={series.isna().sum():,}"
        )

        if series.nunique(dropna=True) <= 10:
            values = series.dropna().unique()
            out.append(f"      Values: {list(values)}")

    for column in december.columns:
        if "date" in column.lower():
            parsed = pd.to_datetime(december[column], errors="coerce")

            if parsed.notna().any():
                out.append("")
                out.append(
                    f"Date column detected: {column}"
                )
                out.append(
                    f"  min={parsed.min()}"
                )
                out.append(
                    f"  max={parsed.max()}"
                )
                out.append(
                    f"  unique={parsed.nunique():,}"
                )

    out.append("")


def environment_info(out: list[str]) -> None:
    out.append("=" * 90)
    out.append("ENVIRONMENT INFORMATION")
    out.append("=" * 90)

    out.append(f"Python: {sys.version}")
    out.append(f"Platform: {platform.platform()}")
    out.append(f"Executable: {sys.executable}")

    try:
        out.append(f"pandas: {pd.__version__}")
    except Exception:
        pass

    try:
        out.append(f"numpy: {np.__version__}")
    except Exception:
        pass

    out.append("")


def main() -> None:
    output: list[str] = []

    output.append("SPOTTER FREIGHT-RATE DATASET PROFILE")
    output.append("=" * 90)
    output.append(
        "Generated by a READ-ONLY profiling script. "
        "No model training or data modification was performed."
    )
    output.append("")

    environment_info(output)

    loaded: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Load files
    # ------------------------------------------------------------------
    for label, path in FILES.items():
        try:
            df = load_csv(path)
            loaded[label] = df

            describe_file(label, path, df, output)

        except Exception as exc:
            output.append("=" * 90)
            output.append(f"FILE: {label}")
            output.append("=" * 90)
            output.append(f"ERROR loading {path}: {type(exc).__name__}: {exc}")
            output.append("")

    # ------------------------------------------------------------------
    # Cross-file analysis
    # ------------------------------------------------------------------
    if "train" in loaded:
        train = loaded["train"]

        if "validation" in loaded:
            validation = loaded["validation"]

            target_analysis(train, validation, output)
            overlap_analysis(train, validation, output)
            correlation_analysis(train, output)

        elif "validation" not in loaded:
            output.append(
                "Validation file unavailable; cross-file analysis skipped."
            )

    if "december" in loaded:
        december_analysis(loaded["december"], output)

    # ------------------------------------------------------------------
    # Overall warnings
    # ------------------------------------------------------------------
    output.append("=" * 90)
    output.append("AUTOMATED WARNINGS / THINGS TO INVESTIGATE")
    output.append("=" * 90)

    warnings_found = False

    for label, df in loaded.items():
        missing_columns = [
            c for c in df.columns
            if df[c].isna().any()
        ]

        if missing_columns:
            warnings_found = True
            output.append(
                f"[{label}] Missing values in: {missing_columns}"
            )

        duplicate_rows = int(df.duplicated().sum())

        if duplicate_rows:
            warnings_found = True
            output.append(
                f"[{label}] Duplicate complete rows: {duplicate_rows:,}"
            )

    if "train" in loaded and "validation" in loaded:
        train = loaded["train"]
        validation = loaded["validation"]

        train_columns = set(train.columns)
        validation_columns = set(validation.columns)

        train_only = sorted(train_columns - validation_columns)
        validation_only = sorted(validation_columns - train_columns)

        if train_only:
            warnings_found = True
            output.append(
                f"[schema] Columns only in training data: {train_only}"
            )

        if validation_only:
            warnings_found = True
            output.append(
                f"[schema] Columns only in validation data: {validation_only}"
            )

    if not warnings_found:
        output.append("No automatic warnings detected.")

    output.append("")
    output.append("=" * 90)
    output.append("END OF PROFILE")
    output.append("=" * 90)

    OUTPUT_FILE.write_text(
        "\n".join(output),
        encoding="utf-8",
    )

    print(f"Profile written to: {OUTPUT_FILE.resolve()}")
    print("Input files were read-only; no dataset files were modified.")


if __name__ == "__main__":
    main()