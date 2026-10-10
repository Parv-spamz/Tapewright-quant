"""
Model registry integration for quantitative and tabular ML models.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from database.repositories.market_data_repo.model_registry_repo import ModelRegistryRepo


def register_model(
    name: str,
    version: str,
    metrics: dict[str, Any],
    path: str,
    repo: Optional[ModelRegistryRepo] = None,
    family: str = "ensemble",
    hyperparameters: Optional[dict[str, Any]] = None,
) -> str:
    """Registers a trained quantitative or ML model into the model registry database.

    Parameters
    ----------
    name : str
        Model identifier name (e.g. 'xgboost_momentum_meta').
    version : str
        Semantic version string (e.g. 'v1.0.0').
    metrics : dict
        Validation and backtest performance metrics (e.g. {'auc': 0.62, 'log_loss': 0.65}).
    path : str
        Filesystem or artifact path where model weights are stored.
    repo : ModelRegistryRepo, optional
        Repository instance. If None, default repository instance is instantiated.
    family : str, default 'ensemble'
        Model architecture family (e.g. 'ensemble', 'deep_learning', 'reinforcement_learning').
    hyperparameters : dict, optional
        Optional model hyperparameters. If None, populated with artifact path.

    Returns
    -------
    str
        Unique registered model_id.
    """
    if repo is None:
        repo = ModelRegistryRepo()

    model_id = f"{name}_{version}_{uuid.uuid4().hex[:6]}"

    hp = hyperparameters.copy() if hyperparameters is not None else {}
    hp["path"] = path
    hp["artifact_path"] = path

    registered_id = repo.register(
        model_id=model_id,
        model_name=name,
        model_family=family,
        version=version,
        hyperparameters=hp,
        metrics=metrics,
    )
    return registered_id
