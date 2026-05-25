# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate synthetic example data for MOSAIC quickstart.

Produces two files in the ``examples/`` directory:

- ``sample_labels.npy`` — binary label vector, 100 samples, balanced classes.
- ``sample_PCA70.npz``  — synthetic PCA-reduced feature matrix, 100×37,
  with keys ``X`` (float32) and ``y`` (matches ``sample_labels.npy``).

The generation is fully deterministic (seed=42). Running this script
again produces byte-identical files.

Usage
-----
From the project root::

    python examples/generate_sample_data.py
"""

from pathlib import Path

import numpy as np

SEED = 42
N_SAMPLES = 100
N_FEATURES = 37  # matches real PCA70 shape
OUTPUT_DIR = Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(SEED)

    # Binary labels — balanced classes
    y = np.array([0] * (N_SAMPLES // 2) + [1] * (N_SAMPLES // 2), dtype=np.int64)

    # Feature matrix — float32 values in [0, 1]
    X = rng.random((N_SAMPLES, N_FEATURES)).astype(np.float32)

    labels_path = OUTPUT_DIR / "sample_labels.npy"
    npz_path = OUTPUT_DIR / "sample_PCA70.npz"

    np.save(labels_path, y)
    np.savez(npz_path, X=X, y=y)

    print(f"Generated: {labels_path}  shape={y.shape}")
    print(f"Generated: {npz_path}  X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
