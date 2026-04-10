import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer

import xgboost as xgb
from catboost import CatBoostRegressor


# -----------------------------
# Metrics / Scorers
# -----------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)


def evaluate_model(y_true, y_pred, name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse_val = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Results")
    print(f"MSE: {mse:.6f} | RMSE: {rmse_val:.6f} | MAE: {mae:.6f} | R^2: {r2:.6f}")


# -----------------------------
# Models + Hyperparameter grids
# -----------------------------
rf_model = RandomForestRegressor(random_state=42)
rf_params = {
    "n_estimators": [100, 200, 400, 600],
    "max_depth": [10, 20, 30, 40]
}

xgb_model = xgb.XGBRegressor(
    tree_method="hist",
    objective="reg:squarederror",
    random_state=42
)
xgb_params = {
    "n_estimators": [500, 1000, 2000, 4000],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "gamma": [0, 0.1, 0.2, 0.3, 0.5]
}

cb_model = CatBoostRegressor(
    random_state=42,
    verbose=0
    # If you really use GPU:
    # task_type="GPU"
)
cb_params = {
    "iterations": [500, 1000, 2000],
    "depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.3],
    "l2_leaf_reg": [3, 5, 10],
    "bagging_temperature": [0.2, 0.6, 1.0]
}

svr_model = SVR()
svr_params = {
    "C": [0.1, 1, 10],
    "epsilon": [0.01, 0.1, 0.2],
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto"]
}


# -----------------------------
# GridSearch + Training
# -----------------------------
rf_grid = GridSearchCV(
    rf_model, rf_params, scoring=rmse_scorer, cv=5, n_jobs=-1, verbose=2
)
xgb_grid = GridSearchCV(
    xgb_model, xgb_params, scoring=rmse_scorer, cv=5, n_jobs=-1, verbose=2
)
cb_grid = GridSearchCV(
    cb_model, cb_params, scoring=rmse_scorer, cv=5, n_jobs=-1, verbose=2
)
svr_grid = GridSearchCV(
    svr_model, svr_params, scoring=rmse_scorer, cv=5, n_jobs=-1, verbose=2
)

# If you use feature-selected matrices per-model, plug them here:
# rf_grid.fit(X_train_rf, y_train)
# xgb_grid.fit(X_train_xgb, y_train)
# cb_grid.fit(X_train_cb, y_train)
# svr_grid.fit(X_train, y_train)

rf_grid.fit(X_train, y_train)
xgb_grid.fit(X_train, y_train)
cb_grid.fit(X_train, y_train)
svr_grid.fit(X_train, y_train)

print("\nBest params:")
print("RF :", rf_grid.best_params_)
print("XGB:", xgb_grid.best_params_)
print("CB :", cb_grid.best_params_)
print("SVR:", svr_grid.best_params_)


# -----------------------------
# Predictions + Evaluation
# -----------------------------
rf_pred = rf_grid.predict(X_test)
xgb_pred = xgb_grid.predict(X_test)
cb_pred = cb_grid.predict(X_test)
svr_pred = svr_grid.predict(X_test)

evaluate_model(y_test, rf_pred, "Random Forest")
evaluate_model(y_test, xgb_pred, "XGBoost")
evaluate_model(y_test, cb_pred, "CatBoost")
evaluate_model(y_test, svr_pred, "SVR")
