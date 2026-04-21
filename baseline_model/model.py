import numpy as np
import cloudpickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """LogisticRegression wrapper ami 0.3 küszöböt használ 0.5 helyett."""
    def __init__(self, estimator, threshold=0.3):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba > self.threshold).astype(int)

    # FL miatt szükséges — a strategy ezeken keresztül olvassa/írja a súlyokat
    @property
    def coef_(self):
        return self.estimator.coef_

    @coef_.setter
    def coef_(self, value):
        self.estimator.coef_ = value

    @property
    def intercept_(self):
        return self.estimator.intercept_

    @intercept_.setter
    def intercept_(self, value):
        self.estimator.intercept_ = value

    @property
    def warm_start(self):
        return self.estimator.warm_start


def get_model():
    def medical_feature_engineering(X):
        hr     = X[:, [0]]
        sbp    = X[:, [3]]
        map_   = X[:, [4]]
        dbp    = X[:, [5]]
        resp   = X[:, [6]]
        bun    = X[:, [15]]
        creat  = X[:, [19]]
        lact   = X[:, [22]]
        wbc    = X[:, [31]]
        age    = X[:, [34]]
        iculos = X[:, [39]]

        shock_index     = hr / (sbp + 1e-6)
        bun_creat_ratio = bun / (creat + 1e-6)
        map_hr_ratio    = map_ / (hr + 1e-6)
        resp_hr_ratio   = resp / (hr + 1e-6)
        pulse_pressure  = sbp - dbp
        lactate_flag    = (lact > 2.0).astype(float)
        elderly_flag    = (age > 65).astype(float)
        icu_long_flag   = (iculos > 72).astype(float)
        wbc_flag        = (wbc > 12.0).astype(float)
        composite_risk  = shock_index + lactate_flag + wbc_flag

        return np.hstack([
            X,
            shock_index, bun_creat_ratio,
            map_hr_ratio, resp_hr_ratio,
            pulse_pressure, lactate_flag,
            elderly_flag, icu_long_flag,
            wbc_flag, composite_risk
        ])

    base_lr = LogisticRegression(
        warm_start=True,
        max_iter=1,
        class_weight={0: 1, 1: 3},
        C=0.05,
        solver="saga",
        tol=1e-3,
    )

    model = Pipeline([
        ("engineering", FunctionTransformer(medical_feature_engineering)),
        ("scaler",      StandardScaler()),
        ("clf",         ThresholdClassifier(estimator=base_lr, threshold=0.3)),
    ])

    model.fit(np.zeros((5, 40)), np.array([0, 1, 0, 1, 0]))
    return model


def get_model_parameters(model):
    return [model.named_steps["clf"].coef_, model.named_steps["clf"].intercept_]


def set_model_parameters(model, parameters):
    model.named_steps["clf"].coef_      = parameters[0]
    model.named_steps["clf"].intercept_ = parameters[1]


def save_model(model, path="final_model.pkl"):
    with open(path, "wb") as f:
        cloudpickle.dump(model, f)


def load_model(path="final_model.pkl"):
    with open(path, "rb") as f:
        return cloudpickle.load(f)