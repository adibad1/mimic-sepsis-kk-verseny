import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
    FunctionTransformer,
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def medical_feature_engineering(X):
    """
    Custom Feature Engineering:
    1. Shock Index = Heart Rate / Systolic Blood Pressure
    2. BUN/Creatinine Ratio = Kidney stress indicator
    """
    # Assuming column indices based on DATA_DICTIONARY.md:
    # HR=1, SBP=3, BUN=16, Creatinine=20
    # We add a small epsilon (1e-6) to prevent division by zero
    hr = X[:, [1]]
    sbp = X[:, [3]]
    bun = X[:, [16]]
    creat = X[:, [20]]

    shock_index = hr / (sbp + 1e-6)
    bun_creat_ratio = bun / (creat + 1e-6)

    return np.hstack([X, shock_index, bun_creat_ratio])


def get_model():
    """
    A rich pipeline containing:
    - Custom Division (Medical Ratios)
    - Multiplication (Interactions)
    - Normalization (StandardScaler)
    - Dimensionality Reduction (PCA)
    - Classification (Logistic Regression)
    """
    # 1. Custom Transformer for Division/Ratios
    feat_eng = FunctionTransformer(medical_feature_engineering)

    # 2. Pipeline definition
    model = Pipeline(
        [
            ("engineering", feat_eng),  # Add Shock Index & BUN/Creat
            (
                "poly",
                PolynomialFeatures(degree=2, interaction_only=True),
            ),  # Multiplications
            ("scaler", StandardScaler()),  # Normalization
            ("clf", LogisticRegression(warm_start=True, max_iter=1)),
        ]
    )

    # Initialize the model with dummy data (40 features)
    # This is required so the internal shapes of PCA and Scaler are set.
    model.fit(np.zeros((5, 40)), np.array([0, 1, 0, 1, 0]))
    return model


def get_model_parameters(model):
    """Extracts weights from the 'clf' step of the pipeline."""
    return [model.named_steps["clf"].coef_, model.named_steps["clf"].intercept_]


def set_model_parameters(model, parameters):
    """Injects weights into the 'clf' step of the pipeline."""
    model.named_steps["clf"].coef_ = parameters[0]
    model.named_steps["clf"].intercept_ = parameters[1]
