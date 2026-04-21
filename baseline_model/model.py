import numpy as np
import cloudpickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_model():
    """
    Returns a pipeline with feature engineering + MLP.
    The scaler is NOT part of the pipeline – it will be fitted locally per client.
    """
    def medical_feature_engineering(X):
        hr    = X[:, [0]]
        sbp   = X[:, [3]]
        map_  = X[:, [4]]
        resp  = X[:, [6]]
        temp  = X[:, [2]]
        lact  = X[:, [22]]
        wbc   = X[:, [31]]
        iculos= X[:, [39]]

        shock_index = hr / (sbp + 1e-6)
        shock_index_sq = np.square(shock_index)
        log_lactate = np.log1p(np.abs(lact))
        hr_squared  = np.square(hr / 100.0)
        lactate_high = (lact > 2.0).astype(float)
        wbc_danger   = ((wbc > 12.0) | (wbc < 4.0)).astype(float)
        hypotension  = (sbp < 90.0).astype(float)
        time_risk = np.sqrt(iculos + 1e-6)

        return np.hstack([
            X,
            shock_index, shock_index_sq, log_lactate,
            hr_squared, lactate_high, wbc_danger,
            hypotension, time_risk
        ])

    model = Pipeline([
        ("engineering", FunctionTransformer(medical_feature_engineering)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(32, 16),   # reduced size → lower cost
            activation="tanh",
            solver="adam",
            learning_rate_init=3e-4,
            batch_size=32,
            alpha=0.001,
            max_iter=1,                    # one epoch per round
            warm_start=False,              # fresh model each round
            random_state=42
        )),
    ])
    return model

def get_model_parameters(model):
    """Extract only MLP weights & biases (not the scaler)."""
    clf = model.named_steps["clf"]
    return clf.coefs_ + clf.intercepts_

def set_model_parameters(model, parameters):
    """Set MLP parameters from aggregated list."""
    clf = model.named_steps["clf"]
    n_layers = len(clf.hidden_layer_sizes) + 1
    clf.coefs_ = parameters[:n_layers]
    clf.intercepts_ = parameters[n_layers:]

def save_model(model, path="final_model.pkl"):
    with open(path, "wb") as f:
        cloudpickle.dump(model, f)

def load_model(path="final_model.pkl"):
    with open(path, "rb") as f:
        return cloudpickle.load(f)