"""Training module for BoneStrengthML pipeline."""

from bonestrength_ml.training.metrics import evaluate_predictions, rmse, rmse_scorer
from bonestrength_ml.training.model_factory import MODEL_REGISTRY, create_model, get_param_grid
from bonestrength_ml.training.splitter import TrainTestData, split_data_for_all_outputs
from bonestrength_ml.training.trainer import TrainingResult, train_single_model

__all__ = [
    "MODEL_REGISTRY",
    "TrainTestData",
    "TrainingResult",
    "create_model",
    "evaluate_predictions",
    "get_param_grid",
    "rmse",
    "rmse_scorer",
    "split_data_for_all_outputs",
    "train_single_model",
]
