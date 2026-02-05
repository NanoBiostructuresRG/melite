# MOSAIC: Modular Multi-Model Selection & Cross-Validation
**Version 1.0.0 – May, 2025. Oviedo**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[![Version](https://img.shields.io/badge/version-v1.0-blue.svg)]()

---

## Description
**MOSAIC** is a lightweight benchmarking toolkit for tabular classification. It applies PCA or UMAP reductions, grid‑searches SVC, Random Forest and XGBoost, evaluates each setup with *N × M* Repeated‑Stratified K‑Fold, retrains the top configuration on the full data set, exports a ready‑to‑deploy `.pkl`, and saves a three‑panel plot (F1, Accuracy, AUC‑ROC) that visualises all *N × M* cross‑validation folds.

---

## Purpose
The primary objective of MOSAIC is to automate the preparation of molecular datasets for cheminformatics workflows and **phase 2** machine learning applications within the computational drug discovery pipeline. The platform enables:
- **Standardized baseline** for comparing tabular models.
- **Facilitate reproducibility** in academic and industrial experiments.
- **Produce final artifacts (.pkl + figures)** ready for development or publication.

---

## Architecture
```text
[raw/.npy, .npz]  →  load_dataset.py  →  model_training.py  →  main.py
                                                   │
                                                   ├─ results.txt / results.csv
                                                   └─ figures/*.png

export_best_model.py  →  re‑fit best model  →  output/*.pkl (ML-model)
```

---

## Project Structure
```text
MOSAIC/Phase 2
│
├── raw/
│   ├── labels.npy                      # Target vector (from phase 1-ML)
│   └── morgan_db_{training_set}.npy    # Original features (from phase 1-ML)
│
├── data/
│   ├── PCA.npz                         # Reduced matrices: X_PCA70, X_PCA85, …
│   └── UMAP.npz                        # Reduced matrices: X_UMAP70, X_UMAP85, …
│
├── config.py                           # Global parameters (paths, seeds, grids, CV)
├── main.py                             # End‑to‑end pipeline
├── load_dataset.py                     # Loads X / y for any reduction and level
├── model_training.py                   # GridSearchCV, cross‑validation, model pick
├── result_manager.py                   # Writes human‑readable logs to TXT
├── export_best_model.py                # CLI: choose row from CSV, retrain, save .pkl + plot
├── plot_metrics.py                     # Generates 1×3 box + jitter plots
│
├── output/
│   ├── results.txt                     # Summary (best model per reduction/level)
│   ├── results.csv                     # Full metric table
│   ├── Model_ML_PCA/UMAP.pkl           # Output - model re‑fit on the whole data set
│   │
│   └── figures/
│        └── ML_PCA/UMAP.png            # Example plot of *N × M* CV folds
│   
└── README.md                           # This file
```

---

## How to Run
From the project root directory, run the following command:

```bash
python main.py                # train all models, write TXT and CSV

python export_best_model.py   # pick best row, retrain, save .pkl + figure
```

---

## Output

The following files will be saved under the `artifacts/` directory:

- `Model_SVC_PCA80.pkl`  

---

## Example Console Output

```text
# Training phase
Running with PCA...
INFO:load_dataset:Labels
Training with PCA85 (level=85).
Running with UMAP...
INFO:load_dataset:Labels loaded:
Training with UMAP85 (level=85).
Final report written to output/results.txt
CSV file written to output/results.csv

# Export phase
$ python export_best_model.py
--------------------------------------------
  reduction_type  level              model_name  f1_macro  accuracy  auc_roc
0            PCA     85                     SVC    0.8336    0.8408   0.8802
1           UMAP     85  RandomForestClassifier    0.7041    0.7097   0.7855


Enter the row number to keep: 0

Training SVC on PCA85 using all available data...

```

---

## Notes

- The PCA<level>.npz and UMAP<level>.npz matrices must be in data/.
- The script automatically detects if AUC-ROC does not apply (multiclass).
- The figures use np.random.seed(42) to make the jitter reproducible.
- Hypergrids can be extended by editing config.py without changing the rest of the code.

---

## Future Extensions

Add the **prediction module** that loads any artefact in `output/artefacts/`,
performs input-validation, and exposes two interfaces:

```text
Batch CLI
python predict.py --model artefacts/Model_SVC_PCA70.pkl \
                  --X new_fps.npy \
                  --out preds.csv
```
```text
Library usage
from mosaic_ml.inference import load_model, predict_proba
probas = predict_proba("artefacts/Model_SVC_PCA70.pkl", new_features)
```
---

## Authors

Developed by **Flavio F. Contreras-Torres** (Tecnológico de Monterrey)  
Oviedo, Spain – May 2025

Co-authors: **Ana C. Murrieta** (Tecnológico de Monterrey)

---

## License
This project is licensed under the terms of the [MIT License](https://github.com/NanoBiostructuresRG/molraptor/blob/main/LICENSE).  
See the LICENSE file for full details.