# Quick Start

The fastest way to try MELITE is with the bundled synthetic example dataset.

## Install

```bash
git clone https://github.com/NanoBiostructuresRG/melite.git
cd melite
python -m pip install -e .
```

## Run a Smoke Benchmark

```bash
melite run --smoke --config examples/example_config.toml
```

Smoke mode uses a small grid and fast cross-validation settings. It is useful
for checking installation and data flow, but it is not benchmark-quality.

## Export a Model Artifact

```bash
melite export --row 0 --csv examples/output/results.csv --outdir examples/output/
```

This retrains the selected model on all available example data and writes a
`.pkl` artifact.

## Predict from Python

```python
import numpy as np
from melite import predict

X_new = np.load("examples/sample_PCA70.npz")["X"]
result = predict("examples/output/Model_SVC_PCA70.pkl", X_new)

print(result["predictions"])    # shape (n_samples,)
print(result["probabilities"])  # shape (n_samples, n_classes)
```

## Expected Workflow

```text
X / y -> melite run -> results.csv -> melite export -> .pkl -> predict()
```
