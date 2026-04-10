def grid_search_rf(X, y, scorer):
    model = RandomForestRegressor(random_state = 42, verbose=10)
    params = {
        'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750],  
        'max_depth': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],  # suggested by chatGPT
    }
    return GridSearchCV(model, params, scoring = scorer, cv = 5, n_jobs=-1).fit(X,y).best_estimator_

def grid_search_xgb(X, y, scorer): 
    model = xgb.XGBRegressor(device='cuda', random_state=42, verbosity=3)
    params = {
        'n_estimators': [i for i in range(100, 8000, 500)],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }
    return GridSearchCV(model, params, scoring = scorer, cv = 5).fit(X, y).best_estimator_

def grid_search_cb(X, y, scorer): 
    model = CatBoostRegressor(task_type='GPU', random_state = 42, verbose=10)
    params = {
           'iterations': [100, 500, 1000],
            'depth': [4, 6, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'l2_leaf_reg': [3, 5, 10],
            'bagging_temperature': [0.2, 0.6, 1.0]
    }
    return GridSearchCV(model, params, scoring=scorer, cv=5).fit(X, y).best_estimator_

def grid_search_pls(X, y, scorer):
    model = PLSRegression()
    params = {'n_components': list(range(2, min(X.shape[1], 95)))}
    return GridSearchCV(model, params, scoring=scorer, cv=5, verbose = 10).fit(X, y).best_estimator_

def grid_search_gpr(X, y, scorer):
    
    # Kernels with explicit bounds
    kernel_rbf_1 = RBF(length_scale=0.1, length_scale_bounds=(1e-2, 10.0))
    kernel_rbf_2 = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0))
    kernel_rbf_3 = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)) + DotProduct() + WhiteKernel(noise_level=1e-3)

    # Pipeline: scaling + GPR
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            random_state=42,
            normalize_y=True,
            optimizer="fmin_l_bfgs_b"
        ))
    ])

    # Parameter grid targeting GPR step
    param_grid = {
        "gpr__kernel": [kernel_rbf_1, kernel_rbf_2, kernel_rbf_3],
        "gpr__alpha": [1e-5, 1e-2, 1e-1]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=3
    )

    # Fit and return the best estimator
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def grid_search_svr(X, y, scorer):
    model = SVR()
    params = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2, 0.5],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3]
    }
    return GridSearchCV(model, params, scoring=scorer, cv=5, n_jobs=-1, verbose = 10).fit(X, y).best_estimator_

def grid_search_mlp(X, y, scorer):
    model = MLPRegressor(random_state=42, max_iter=1000)
    params = {
        'hidden_layer_sizes': [(50,), (100,)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001],
        'alpha': [0.0001, 0.001]
    }
    return GridSearchCV(model, params, scoring=scorer, cv=5, n_jobs=-1, verbose = 10).fit(X, y).best_estimator_
