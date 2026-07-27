"""Neural networks as catalogue models, so every verb already works on them.

Design Principles
-----------------
- **A training loop is a tool, not a framework.** michi writes the epochs,
  the optimiser, the batching, and the early stopping — the code an engineer
  otherwise retypes for the fortieth time — and hands back something with
  ``fit`` and ``predict``. Nothing about your project has to change to use it,
  which is the whole difference between a toolbox and a framework.
- **Architecture is a menu, not a default with an opinion.** Layer sizes,
  learning rate, dropout, and patience are hyperparameters like any other:
  visible in ``tune --list-space``, settable from a recipe of parameters, and
  never chosen for you on the grounds that michi "knows" what suits your data.
- **Two rungs, not ten.** ``mlp`` needs nothing beyond scikit-learn and works
  the moment michi is installed. ``torch-mlp`` is a real PyTorch loop for
  people who want the GPU, the schedule, and the checkpoint. A per-architecture
  zoo would be a framework by another name.
- **Honest about what a net is worth.** A network trains inside the same
  cross-validation as everything else, is compared with the same corrected
  *t*-test, and sits under the same dummy baseline. michi never presents "we
  used deep learning" as a result.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_mlp", "build_torch_mlp"]


def build_mlp(task: str, seed: int) -> Any:
    """A multi-layer perceptron on scikit-learn's own training loop.

    Needs no extra dependency, which matters more than it sounds: the first
    neural network someone tries should not require an install, a CUDA
    version, and an afternoon.
    """
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    shared: dict[str, Any] = {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "max_iter": 500,
        # Early stopping is on because a net that trains to convergence on a
        # small tabular set has memorised it. This is a mechanic, not a
        # judgement: it is the same reason `bench` cross-validates at all.
        "early_stopping": True,
        "n_iter_no_change": 15,
        "random_state": seed,
    }
    if task == "regression":
        return MLPRegressor(**shared)
    return MLPClassifier(**shared)


def build_torch_mlp(task: str, seed: int) -> Any:
    """A PyTorch network wrapped so that ``fit``/``predict`` is all michi sees.

    This is the loop an engineer writes over and over: epochs, batches, Adam,
    a validation split, early stopping, and restoring the best weights. It is
    written once here so it is not written again — and because it satisfies
    the estimator protocol, ``bench``, ``tune``, ``fit``, and ``predict`` all
    work on it with no special case anywhere in michi.
    """
    from michi.core.errors import RunError, install_hint

    try:
        import torch  # noqa: F401
    except ImportError as err:
        msg = (
            "torch-mlp needs PyTorch, which michi does not install by default "
            f"because it is large and platform-specific. {install_hint('torch')}"
        )
        raise RunError(msg) from err

    return _torch_estimator()(task=task, seed=seed)


def _torch_estimator() -> Any:
    """Build the estimator class lazily, so importing michi never needs torch."""
    import numpy as np
    import torch
    from sklearn.base import BaseEstimator
    from torch import nn

    class TorchMLP(BaseEstimator):  # type: ignore[misc]
        """A feed-forward network with the training loop michi writes for you.

        Parameters
        ----------
        hidden
            Width of each hidden layer.
        dropout
            Dropout probability between layers.
        learning_rate
            Adam's initial step size.
        epochs
            Maximum passes over the data; early stopping usually ends sooner.
        batch_size
            Rows per gradient step.
        patience
            Epochs without validation improvement before stopping.
        device
            ``"auto"`` picks CUDA, then Apple MPS, then CPU.
        """

        def __init__(
            self,
            task: str = "classification",
            hidden: tuple[int, ...] = (256, 128),
            dropout: float = 0.1,
            learning_rate: float = 1e-3,
            epochs: int = 200,
            batch_size: int = 256,
            patience: int = 15,
            device: str = "auto",
            seed: int = 0,
        ) -> None:
            self.task = task
            self.hidden = hidden
            self.dropout = dropout
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.patience = patience
            self.device = device
            self.seed = seed

        # -- the parts michi's other verbs rely on ----------------------

        def fit(self, X: Any, y: Any) -> Any:
            """Train, holding out a slice to stop on."""
            from sklearn.model_selection import train_test_split

            torch.manual_seed(self.seed)
            device = self._device()

            features = np.asarray(X, dtype="float32")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            targets, out_features = self._encode_targets(y)

            train_x, valid_x, train_y, valid_y = train_test_split(
                features,
                targets,
                test_size=0.15,
                random_state=self.seed,
                stratify=targets if self.task == "classification" else None,
            )

            self.module_ = self._build(features.shape[1], out_features).to(device)
            loss_fn = (
                nn.CrossEntropyLoss() if self.task == "classification" else nn.MSELoss()
            )
            optimiser = torch.optim.Adam(
                self.module_.parameters(), lr=self.learning_rate
            )

            train_tensors = self._tensors(train_x, train_y, device)
            valid_features = torch.tensor(valid_x, device=device)
            valid_targets = self._target_tensor(valid_y, device)

            best = float("inf")
            best_state = {
                key: value.detach().clone()
                for key, value in self.module_.state_dict().items()
            }
            waited = 0

            for _ in range(self.epochs):
                self.module_.train()
                for batch_x, batch_y in self._batches(train_tensors):
                    optimiser.zero_grad()
                    loss = loss_fn(self.module_(batch_x), batch_y)
                    loss.backward()
                    optimiser.step()

                self.module_.eval()
                with torch.no_grad():
                    validation = float(
                        loss_fn(self.module_(valid_features), valid_targets)
                    )
                # Restoring the best weights rather than the last is the
                # difference between early stopping and merely stopping early.
                if validation < best - 1e-5:
                    best = validation
                    best_state = {
                        key: value.detach().clone()
                        for key, value in self.module_.state_dict().items()
                    }
                    waited = 0
                else:
                    waited += 1
                    if waited >= self.patience:
                        break

            self.module_.load_state_dict(best_state)
            self.module_.eval()
            return self

        def predict(self, X: Any) -> Any:
            """Predict labels or values."""
            scores = self._forward(X)
            if self.task == "classification":
                return self.classes_[scores.argmax(axis=1)]
            return scores.reshape(-1)

        def predict_proba(self, X: Any) -> Any:
            """Class probabilities, so `eval` can calibrate and rank."""
            if self.task != "classification":
                msg = "predict_proba is meaningless for a regression network"
                raise AttributeError(msg)
            scores = torch.tensor(self._forward(X))
            return torch.softmax(scores, dim=1).numpy()

        # -- internals --------------------------------------------------

        def _forward(self, X: Any) -> Any:
            features = np.asarray(X, dtype="float32")
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            with torch.no_grad():
                tensor = torch.tensor(features, device=self._device())
                return self.module_(tensor).cpu().numpy()

        def _build(self, in_features: int, out_features: int) -> Any:
            layers: list[Any] = []
            width = in_features
            for size in self.hidden:
                layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(self.dropout)]
                width = size
            layers.append(nn.Linear(width, out_features))
            return nn.Sequential(*layers)

        def _encode_targets(self, y: Any) -> tuple[Any, int]:
            values = np.asarray(y)
            if self.task == "regression":
                return values.astype("float32"), 1
            self.classes_ = np.unique(values)
            lookup = {label: index for index, label in enumerate(self.classes_)}
            encoded = np.array([lookup[item] for item in values], dtype="int64")
            return encoded, len(self.classes_)

        def _target_tensor(self, values: Any, device: Any) -> Any:
            if self.task == "classification":
                return torch.tensor(values, dtype=torch.long, device=device)
            return torch.tensor(values, dtype=torch.float32, device=device).reshape(
                -1, 1
            )

        def _tensors(self, features: Any, targets: Any, device: Any) -> tuple[Any, Any]:
            return (
                torch.tensor(features, device=device),
                self._target_tensor(targets, device),
            )

        def _batches(self, tensors: tuple[Any, Any]) -> Any:
            features, targets = tensors
            order = torch.randperm(len(features), device=features.device)
            for start in range(0, len(order), self.batch_size):
                index = order[start : start + self.batch_size]
                yield features[index], targets[index]

        def _device(self) -> Any:
            if self.device != "auto":
                return torch.device(self.device)
            if torch.cuda.is_available():
                return torch.device("cuda")
            if (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            return torch.device("cpu")

    return TorchMLP
