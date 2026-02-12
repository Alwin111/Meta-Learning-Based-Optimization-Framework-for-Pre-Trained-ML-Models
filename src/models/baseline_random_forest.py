from sklearn.ensemble import RandomForestClassifier

def get_model(n_estimators=100, max_depth=None, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    return model
