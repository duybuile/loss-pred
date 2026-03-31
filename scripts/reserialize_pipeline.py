#!/usr/bin/env python
"""
Re-serializes app/artifacts/feature_pipeline.pkl so transformer classes
are stored with module path 'app.feature_engineering.transformers' instead
of '__main__' (the path produced when pickling from a notebook).

Run this after re-training the model in notebooks/modelling.ipynb:
    uv run python scripts/reserialize_pipeline.py
"""
from __future__ import annotations

import io
import pickle
import sys
from pathlib import Path

# Ensure app package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import app.feature_engineering  # noqa: F401 — registers classes for unpickling


class _ModuleRemapper(pickle.Unpickler):
    """Unpickler that remaps __main__ class lookups to app.feature_engineering.transformers."""

    def find_class(self, module: str, name: str):
        if module == "__main__":
            module = "app.feature_engineering.transformers"
        return super().find_class(module, name)


def reserialize(path: Path) -> None:
    raw = path.read_bytes()
    obj = _ModuleRemapper(io.BytesIO(raw)).load()
    with path.open("wb") as f:
        pickle.dump(obj, f)
    print(f"Re-serialized {path}")


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    reserialize(repo_root / "app/artifacts/feature_pipeline.pkl")
