from sklearn.linear_model import LogisticRegression


def get_baseline_model():
    """
    Returns a Logistic Regression model initialized with a generic state.
    This structure matches the weights expected by the pre-deployed hospital clients.
    """
    model = LogisticRegression(warm_start=True, max_iter=1, random_state=42)
    return model
