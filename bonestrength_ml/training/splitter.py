"""Data splitting utilities based on configuration."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from bonestrength_ml.config import TrainTestSplitConfig


@dataclass
class TrainTestData:
    """Container for train/test split data for a single output."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    output_name: str


def split_data_for_all_outputs(
    X: pd.DataFrame,
    y: pd.DataFrame,
    split_config: TrainTestSplitConfig,
    random_state: int = 42,
) -> dict[str, TrainTestData]:
    """Split data for all output columns using consistent indices.

    Ensures train/test indices are the same across all outputs so models
    are trained and evaluated on the same samples.

    Args:
        X: Feature DataFrame with input columns.
        y: Target DataFrame with output columns.
        split_config: TrainTestSplitConfig from config.
        random_state: Random seed for reproducibility.

    Returns:
        Dict mapping output_name -> TrainTestData.
    """
    # Get indices for consistent split across all outputs
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=split_config.train_fraction,
        random_state=random_state,
        shuffle=True,
    )

    result: dict[str, TrainTestData] = {}
    for output_name in y.columns:
        result[output_name] = TrainTestData(
            X_train=X.iloc[train_idx].reset_index(drop=True),
            X_test=X.iloc[test_idx].reset_index(drop=True),
            y_train=y[output_name].iloc[train_idx].reset_index(drop=True),
            y_test=y[output_name].iloc[test_idx].reset_index(drop=True),
            output_name=output_name,
        )

    return result
