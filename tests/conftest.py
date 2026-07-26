"""Shared fixtures.

Every fixture is generated deterministically and written to a temporary
directory, so the suite is fully offline, leaves nothing behind, and produces
identical data on every platform. Values are constructed so that each planted
problem is unambiguous — no finding should fire by numerical coincidence.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

__all__ = [
    "classification_data",
    "messy_csv",
    "messy_frame",
    "regression_data",
    "tidy_csv",
    "tidy_frame",
]

_ROWS = 120


@pytest.fixture(scope="session")
def messy_frame() -> pd.DataFrame:
    """A dataset carrying one deliberate instance of each detectable problem."""
    index = list(range(_ROWS))
    purchased = [1 if value % 12 == 0 else 0 for value in index]

    return pd.DataFrame(
        {
            # Clean numeric feature.
            "age": [20 + (value * 7) % 45 for value in index],
            # Moderate missingness.
            "salary": [
                None if value % 9 == 0 else 30_000 + (value * 137) % 50_000
                for value in index
            ],
            # Mostly missing.
            "cabin": [f"C{value}" if value % 8 == 0 else None for value in index],
            # Entirely missing.
            "notes": [None] * _ROWS,
            # Single value throughout.
            "country": ["JP"] * _ROWS,
            # Exact duplicate of country.
            "country_copy": ["JP"] * _ROWS,
            # Distinct in every row.
            "record_id": [f"id-{value:04d}" for value in index],
            # Strong right skew with extreme outliers.
            "fare": [
                5.0 + (value % 10) if value < _ROWS - 3 else 10_000.0 for value in index
            ],
            # Numbers pandas will not auto-convert, because of separators.
            "amount_text": [f"{1000 + value:,}" for value in index],
            # Dates as text, on a cycle that cannot align with the target.
            "signup_date": [f"2024-{(value % 11) + 1:02d}-15" for value in index],
            # Perfectly correlated with age.
            "age_months": [(20 + (value * 7) % 45) * 12 for value in index],
            # Encodes the label directly: substantial groups, full coverage.
            "outcome_code": [f"class-{value}" for value in purchased],
            # Imbalanced binary target.
            "purchased": purchased,
        }
    )


@pytest.fixture(scope="session")
def messy_csv(
    tmp_path_factory: pytest.TempPathFactory, messy_frame: pd.DataFrame
) -> Path:
    """The messy dataset written to a CSV file."""
    path = tmp_path_factory.mktemp("messy") / "messy.csv"
    messy_frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def tidy_frame() -> pd.DataFrame:
    """A clean dataset that should produce no high-severity findings."""
    rows = 200
    # The label deliberately cycles on a period coprime with every feature's,
    # so no column accidentally becomes a perfect predictor of it.
    return pd.DataFrame(
        {
            "feature_a": [value % 37 for value in range(rows)],
            "feature_b": [(value * 3) % 11 for value in range(rows)],
            "group": ["north", "south", "east", "west"] * (rows // 4),
            "label": [1 if (value * 7 + 3) % 5 < 2 else 0 for value in range(rows)],
        }
    )


@pytest.fixture(scope="session")
def tidy_csv(
    tmp_path_factory: pytest.TempPathFactory, tidy_frame: pd.DataFrame
) -> Path:
    """The clean dataset written to a CSV file."""
    path = tmp_path_factory.mktemp("tidy") / "tidy.csv"
    tidy_frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def classification_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A trained sklearn pipeline and a held-out test set, as files."""
    import joblib
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    rng = np.random.default_rng(0)
    rows = 600
    age = rng.normal(45, 14, rows).clip(18, 90)
    income = rng.lognormal(10.6, 0.5, rows)
    region = rng.choice(["north", "south", "east", "west"], rows)
    logit = -0.4 + 0.05 * (age - 45) + 1.2 * (np.log(income) - 10.6)
    label = (rng.random(rows) < 1 / (1 + np.exp(-logit))).astype(int)
    frame = pd.DataFrame(
        {"age": age, "income": income, "region": region, "churned": label}
    )

    split = rows // 2
    train, test = frame.iloc[:split], frame.iloc[split:]
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]),
                ["age", "income"],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["region"]),
        ]
    )
    model = Pipeline(
        [
            ("pre", preprocessor),
            ("rf", RandomForestClassifier(n_estimators=40, random_state=0)),
        ]
    )
    model.fit(train.drop(columns="churned"), train["churned"])

    directory = tmp_path_factory.mktemp("classification")
    model_path = directory / "model.pkl"
    data_path = directory / "test.csv"
    joblib.dump(model, model_path)
    test.to_csv(data_path, index=False)
    return model_path, data_path


@pytest.fixture(scope="session")
def regression_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A trained regressor and a held-out test set, as files."""
    import joblib
    import numpy as np
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(1)
    rows = 600
    feature_a = rng.normal(0, 1, rows)
    feature_b = rng.normal(5, 2, rows)
    value = 3.0 * feature_a - 1.5 * feature_b + rng.normal(0, 1, rows)
    frame = pd.DataFrame({"a": feature_a, "b": feature_b, "value": value})

    split = rows // 2
    train, test = frame.iloc[:split], frame.iloc[split:]
    model = Ridge().fit(train[["a", "b"]], train["value"])

    directory = tmp_path_factory.mktemp("regression")
    model_path = directory / "ridge.joblib"
    data_path = directory / "test.csv"
    joblib.dump(model, model_path)
    test.to_csv(data_path, index=False)
    return model_path, data_path
