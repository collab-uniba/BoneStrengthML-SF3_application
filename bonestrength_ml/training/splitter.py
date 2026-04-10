"""Data splitting utilities based on configuration."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

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
    groups: np.ndarray | None = None,
) -> dict[str, TrainTestData]:
    """Split data for all output columns using consistent indices.

    Ensures train/test indices are the same across all outputs so models
    are trained and evaluated on the same samples.

    Supports two methods:
    - "random": simple random split of individual rows.
    - "grouped": group-aware split that keeps all rows belonging to
      the same group (e.g., same patient) in either train or test,
      never both. Requires the ``groups`` parameter.

    Note: when method is "grouped", ``train_fraction`` controls the
    fraction of *groups*, not individual samples. The actual sample
    ratio may differ depending on group sizes.

    Args:
        X: Feature DataFrame with input columns.
        y: Target DataFrame with output columns.
        split_config: TrainTestSplitConfig from config.
        random_state: Random seed for reproducibility.
        groups: Integer array of group labels (one per row). Required
            when split_config.method is "grouped".

    Returns:
        Dict mapping output_name -> TrainTestData.
    """
    if split_config.method == "grouped":
        train_idx, test_idx = _grouped_split(X, groups, split_config, random_state)
    else:
        # Original random split behavior
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


def _grouped_split(
    X: pd.DataFrame,
    groups: np.ndarray | None,
    split_config: TrainTestSplitConfig,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform group-aware train/test split using GroupShuffleSplit.

    Args:
        X: Feature DataFrame.
        groups: Integer array of group labels (one per row).
        split_config: Split configuration with train_fraction.
        random_state: Random seed.

    Returns:
        Tuple of (train_indices, test_indices) as numpy arrays.

    Raises:
        ValueError: If groups is None.
    """
    if groups is None:
        raise ValueError("Groups must be provided when split method is 'grouped'")

    splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=split_config.train_fraction,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(X, groups=groups))
    return train_idx, test_idx
