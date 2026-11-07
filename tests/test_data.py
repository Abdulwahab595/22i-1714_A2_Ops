from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metrics import choose_threshold


def test_fixed_splits_have_disjoint_hashes():
    root = Path("data/splits")
    if not (root / "train.csv").exists():
        pytest.skip("Audit-generated split manifests are not present in this checkout")
    frames = [pd.read_csv(root / f"{name}.csv") for name in ["train", "validation", "test"]]
    hashes = [set(frame["sha256"]) for frame in frames]
    assert all(not (hashes[i] & hashes[j]) for i in range(3) for j in range(i + 1, 3))
    assert all(set(frame["label"]) == {0, 1} for frame in frames)


def test_threshold_policy_prices_real_false_accusations_twice():
    labels, scores = np.array([0, 0, 1, 1]), np.array([0.2, 0.8, 0.7, 0.9])
    threshold, policy = choose_threshold(labels, scores)
    assert 0.8 < threshold <= 0.9
    assert policy["real_called_ai"] == 0
    assert policy["ai_called_real"] == 1
