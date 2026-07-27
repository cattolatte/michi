"""Data preparation, compiled from a michi recipe.

Authored against data/customers.csv on 2026-07-27.

This file is yours. It imports pandas and scikit-learn, and nothing
else — michi is not a runtime dependency of the code it writes.

Two pieces, because they carry different risks:

* ``prepare`` applies the 4 deterministic step(s):
  dropping, deduplicating, casting, clipping. These depend only on the
  row in front of them, so they are safe to run on any data at any time.

* ``build_pipeline`` returns an sklearn Pipeline for the 1
  fitted step(s): imputation, encoding, scaling. These *learn* from the
  data they see, so they belong inside cross-validation, where they
  cannot observe the fold they will be scored on.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the deterministic cleaning steps.

    Returns a new frame; the input is never modified.
    """
    frame = frame.copy()

    # requested with --drop
    # Drop 6 column(s).
    frame = frame.drop(
        columns=[
            "notes",
            "country",
            "country_copy",
            "record_id",
            "outcome_code",
            "age_months",
        ],
        errors="ignore",
    )

    # Parse 1 column(s) as numbers, stripping
    # separators and currency marks. Unparseable values become missing.
    for column in ["amount_text"]:
        cleaned = frame[column].astype(str).str.replace(r"[,\s$€£¥%_]", "", regex=True)
        frame[column] = pd.to_numeric(cleaned, errors="coerce")

    # Parse 1 column(s) as timestamps.
    for column in ["signup_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce", format="mixed")

    # Clip 1 column(s) to the 1%–99% range.
    # Note: the bounds come from whatever data is passed in.
    for column in ["fare"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = values.clip(
            lower=values.quantile(0.01),
            upper=values.quantile(0.99),
        )

    return frame


def build_pipeline() -> ColumnTransformer:
    """Build the transformer for steps that learn from data.

    Fit this inside your cross-validation, never on the whole dataset:
    an imputer fitted on all rows has already seen your test fold.
    """
    return ColumnTransformer(
        transformers=[
            (
                "impute_0",
                SimpleImputer(strategy="median"),
                ["salary"],
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


if __name__ == "__main__":
    # Minimal end-to-end use. Replace the model with your own.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    raw = pd.read_csv("data.csv")
    frame = prepare(raw)
    labels = frame.pop("purchased")

    model = Pipeline(
        [
            ("prepare", build_pipeline()),
            ("model", RandomForestClassifier(random_state=0)),
        ]
    )
    scores = cross_val_score(model, frame, labels, cv=5)
    print(f"{scores.mean():.4f} ± {scores.std():.4f}")
