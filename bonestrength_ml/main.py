import vv4ml as vv

from bonestrength_ml.data_loading import load_and_validate_data, get_input_columns, get_output_columns
from bonestrength_ml.training import prepare_train_test_split, load_best_run

output = "maxStrain_11"
tracking_uri = "file:///home/cittian/Documenti/BoneStrengthML-SF3_application_personal/mlruns"

bs_conf = vv.load_config("../config/BoneStrengthML.yml")
n_repetitions = bs_conf.test_configurations.verification.numerical_error.sets
convergence_rows = 200
threshold = bs_conf.gate_thresholds.verification.numerical_error[0].value
df = load_and_validate_data(bs_conf)

x = df[get_input_columns(bs_conf)]
y = df[get_output_columns(bs_conf)]

data = prepare_train_test_split(x, y, bs_conf)[output]

x_train = x
y_train = y[output]


best_run = load_best_run(
    output,
    tracking_uri=tracking_uri,
    run_id="m-66516a1e65b0498eb8098cbbf6e94ec0",
)
model_type = best_run.model_type
best_params = best_run.best_params

x_eval = data.X_test
result = vv.run_numerical_error_test(
    x_train=x_train,
    y_train=y_train,
    x_eval=x_eval,
    model_type=model_type,
    best_params=best_params,
    n_repetitions=n_repetitions,
    convergence_rows=convergence_rows,
    threshold=threshold,
    output_name=output,
)

print(result)
