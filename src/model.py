"""
GBDT Entity Pair Classifier Module for Amazon ML Challenge 2026.
Supports GPU acceleration (RTX 5070 / CUDA) and CPU training with XGBoost.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import DEFAULT_CONFIG, PipelineConfig

logger = logging.getLogger("model")


class EntityPairClassifier:
    """Pairwise match classification model powered by XGBoost."""

    def __init__(
        self,
        config: PipelineConfig = DEFAULT_CONFIG,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        device: str = "cuda",
        random_state: int = 42,
    ):
        self.config = config
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.device = device
        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []

    def _init_model(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method="hist",
            device=self.device,
            random_state=self.random_state,
            eval_metric="logloss",
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "EntityPairClassifier":
        """Trains XGBoost pair classifier."""
        self.feature_names = list(X_train.columns)
        self.model = self._init_model()

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        logger.info(f"Training XGBoost classifier on device='{self.device}' with {len(X_train):,} pairs...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        logger.info("Model training completed.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts match probabilities for pair features."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        # Ensure column alignment
        X_aligned = X[self.feature_names]
        probs = self.model.predict_proba(X_aligned)[:, 1]
        return probs

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns dict mapping feature name -> importance score."""
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(self.feature_names, importances)}

    def save(self, filepath: Union[str, Path]):
        """Saves model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix == ".json":
            if self.model is not None:
                self.model.save_model(str(filepath))
        else:
            joblib.dump(self, filepath)
        logger.info(f"Saved model to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EntityPairClassifier":
        """Loads model from disk."""
        filepath = Path(filepath)
        if filepath.suffix == ".json":
            obj = cls()
            obj.model = obj._init_model()
            obj.model.load_model(str(filepath))
            return obj
        else:
            return joblib.load(filepath)
