

from sklearn.ensemble import RandomForestClassifier


def get_model(
    n_estimators=120,
    max_depth=20,
    random_state=42,
    n_jobs=-1
):
    """
    Returns a configured Random Forest model.
    This function makes the model reusable across the framework.
    """

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )
