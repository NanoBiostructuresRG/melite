import os
import random
import numpy as np
from sklearn.model_selection import ParameterGrid

class Config:
    def __init__(self):

        self.PATHS = {
            "INPUT": "raw/",
            "DATASET": "data/",
            "OUTPUT": "output/"
        }
        for _p in self.PATHS.values():
            os.makedirs(_p, exist_ok=True)

        self.RESULTS_FILE = os.path.join(self.PATHS["OUTPUT"], "results.txt")
        self.RANDOM_STATE = 42
        self.REDUCTION_LEVELS = [70, 75, 80, 85, 90, 95]  

        self.DIMENSIONALITY_REDUCTION = {
            'PCA': {'LEVELS': [95, 90], 'OUTLIER': True, 'FEATURES': 5},
            'UMAP': {'LEVELS': [95, 90], 'METRICS': ["jaccard"]}
        }

        # CROSS-VALIDATION (RepeatedStratifiedKFold)
        self.CV_CONFIG = {
            "n_splits": 10,      
            "n_repeats": 5,      
            "random_state": self.RANDOM_STATE
        }

        # CONDITIONAL PARAM GRID IN GRIDSEARCHCV
        self.PARAM_GRID = [
            #{
            #    "model": ["svc"],
            #    "kernel": ["linear"],
            #    "C": [0.1, 1, 10],
            #},
            {
                "model": ["svc"],
                "kernel": ["poly"],
                "C": [0.01, 0.1, 1, 10],
                "coef0": [0.0, 0.1, 0.02, 0.6, 0.8, 1],
                "gamma": [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2],
                "degree": [3, 4, 5],
            },
            {
                "model": ["svc"],
                "kernel": ["rbf"],
                "C": [0.01, 0.02, 0.1, 0.02, 1, 2, 10, 20],
                "gamma": [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2],
            },
            {
                "model": ["rf"],
                "n_estimators": [200, 400, 800],
                "max_depth": [None, 10, 20, 30, 40],
                "max_features": ["sqrt", "log2"],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
            {
                "model": ["xgb"],
                "n_estimators": [300, 400, 600],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [4, 6, 8],
                "subsample": [0.7, 0.85],
                "colsample_bytree": [0.7, 1.0],
                "gamma": [0, 0.01, 1, 5],
                "reg_alpha": [0, 0.5],
                "reg_lambda": [1, 2],
            }
        ]
        # ------------------------------------------------------------------ #
        self._set_seeds() 
        self.PARAM_GRID_BY_MODEL = self._group_param_grid_by_model()

    def _set_seeds(self):
        random.seed(self.RANDOM_STATE)
        np.random.seed(self.RANDOM_STATE)

    def _group_param_grid_by_model(self):
        grids = {}
        for entry in self.PARAM_GRID:
            model = entry["model"][0]
            grids.setdefault(model, []).append(
                {k: v for k, v in entry.items() if k != "model"}
            )
        return {m: ParameterGrid(g) for m, g in grids.items()}
    

    def get_cv_config(self):
        return self.CV_CONFIG

    def get_param_grid(self, model: str):
        return self.PARAM_GRID_BY_MODEL[model]
    