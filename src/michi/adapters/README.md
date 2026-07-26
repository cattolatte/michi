# michi.adapters

**Planned — v0.2 (with `eval`). Not implemented yet; this stub is intentional.**

Model loading behind a deliberately narrow policy (PLAN.md §11, LOCKED):

1. sklearn-compatible estimators via pickle/joblib;
2. everything else via the `--model mymodule:obj` protocol — any object with
   `predict(X)` (optionally `predict_proba`).

That protocol covers PyTorch, TensorFlow, ONNX, and custom models without
per-framework loaders. Native format loaders require an ADR and an extra;
"load any `.pt` file" is never promised.
