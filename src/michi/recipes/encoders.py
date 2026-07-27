"""Target encoding, cross-fitted, with the arithmetic visible.

Design Principles
-----------------
- **Reproducible on every supported scikit-learn.** ``sklearn.preprocessing``
  gained a ``TargetEncoder`` in 1.3, but the parameter that makes it
  deterministic was deprecated in 1.9 and is removed in 1.11, and the
  replacement needs a newer floor than michi's. Two identical runs must give
  two identical numbers, so michi owns thirty lines rather than a dependency
  bump that would drop users.
- **Out-of-fold or nothing.** Encoding a row with a mean that included that
  row's own label is the single most common silent leak in tabular ML, and it
  produces a model that scores beautifully and generalises not at all. Every
  training row here is encoded by a fold that did not contain it.
- **Legible enough to be exported.** ``michi export`` writes this class into
  the file it generates, so the reader can see exactly what the encoding did.
  That rules out anything clever.
- **Unseen categories fall back to the prior**, never to zero or to NaN: a
  category the training folds never saw carries no information about the
  target, and the global mean is what "no information" looks like.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["DEFAULT_SMOOTHING", "build_target_encoder", "encode_frame"]

DEFAULT_SMOOTHING = 20.0
"""Weight given to the global mean when a category is small.

A category seen twice should not be trusted the way one seen two thousand
times is. The encoded value is a weighted average of the category's own mean
and the global mean, and this is the weight on the global side — read it as
"how many observations of evidence it takes before the category speaks for
itself".
"""


def _build() -> Any:
    """Construct the encoder class lazily, so importing michi stays cheap."""
    import numpy as np
    import pandas as pd
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.model_selection import KFold

    class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):  # type: ignore[misc]
        """Replace each category with the target's mean for that category.

        Fitted values are computed out of fold: every training row is encoded
        using folds that did not contain it, so the encoding cannot memorise
        the row's own label.

        Parameters
        ----------
        smoothing
            Weight on the global mean for small categories.
        n_splits
            Folds used for the out-of-fold pass.
        random_state
            Seed for the fold split. Two runs with the same seed produce the
            same numbers; that is the whole reason this class exists.
        """

        def __init__(
            self,
            smoothing: float = DEFAULT_SMOOTHING,
            n_splits: int = 5,
            random_state: int = 0,
        ) -> None:
            self.smoothing = smoothing
            self.n_splits = n_splits
            self.random_state = random_state

        def _means(self, column: Any, y: Any, prior: float) -> Any:
            grouped = y.groupby(column.astype("object"), dropna=False)
            counts = grouped.count()
            sums = grouped.sum()
            return (sums + self.smoothing * prior) / (counts + self.smoothing)

        def fit(self, X: Any, y: Any) -> Any:
            """Learn the full-data mapping, used when transforming new rows."""
            frame = pd.DataFrame(X).reset_index(drop=True)
            target = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
            self.prior_ = float(target.mean())
            self.mappings_ = {
                name: self._means(frame[name], target, self.prior_)
                for name in frame.columns
            }
            self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
            return self

        def transform(self, X: Any) -> Any:
            """Encode with the full-data mapping — for data not used to fit."""
            frame = pd.DataFrame(X).reset_index(drop=True)
            out = np.empty((len(frame), frame.shape[1]), dtype=float)
            for position, name in enumerate(frame.columns):
                mapping = self.mappings_[name]
                encoded = frame[name].astype("object").map(mapping)
                out[:, position] = encoded.fillna(self.prior_).to_numpy(dtype=float)
            return out

        def fit_transform(self, X: Any, y: Any = None, **kwargs: Any) -> Any:
            """Fit, and encode the training rows out of fold.

            The two halves differ on purpose. ``transform`` uses everything,
            because new rows were not part of fitting. Training rows must not
            see themselves, so each is encoded by a mapping built from the
            other folds.
            """
            if y is None:
                msg = "target encoding needs the target; pass y to fit_transform"
                raise ValueError(msg)

            frame = pd.DataFrame(X).reset_index(drop=True)
            target = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
            self.fit(frame, target)

            out = np.full((len(frame), frame.shape[1]), self.prior_, dtype=float)
            splits = min(self.n_splits, len(frame))
            if splits < 2:
                # Too few rows to hold anything out; the prior is the only
                # honest answer, and it is what `out` already contains.
                return out

            folds = KFold(n_splits=splits, shuffle=True, random_state=self.random_state)
            for train_index, test_index in folds.split(frame):
                inner_y = target.iloc[train_index]
                prior = float(inner_y.mean())
                for position, name in enumerate(frame.columns):
                    mapping = self._means(frame[name].iloc[train_index], inner_y, prior)
                    encoded = frame[name].iloc[test_index].astype("object").map(mapping)
                    out[test_index, position] = encoded.fillna(prior).to_numpy(
                        dtype=float
                    )
            return out

        def get_feature_names_out(self, input_features: Any = None) -> Any:
            """Names are unchanged: one encoded column per input column."""
            if input_features is not None:
                return np.asarray(input_features, dtype=object)
            return self.feature_names_in_

    return OutOfFoldTargetEncoder


def build_target_encoder(
    smoothing: float = DEFAULT_SMOOTHING,
    n_splits: int = 5,
    random_state: int = 0,
) -> Any:
    """Build an out-of-fold target encoder.

    A function rather than a module-level class so that importing
    :mod:`michi.recipes` never pulls in scikit-learn, which the package
    guarantees.

    Examples
    --------
    >>> encoder = build_target_encoder(smoothing=10.0)
    >>> encoder.smoothing
    10.0
    """
    return _build()(smoothing=smoothing, n_splits=n_splits, random_state=random_state)


def encode_frame(
    frame: pd.DataFrame,
    columns: list[str],
    labels: pd.Series,
    *,
    smoothing: float = DEFAULT_SMOOTHING,
    seed: int = 0,
) -> Any:
    """Out-of-fold encode `columns` of `frame` against `labels`."""
    encoder = build_target_encoder(smoothing=smoothing, random_state=seed)
    return encoder.fit_transform(frame[columns], labels)
