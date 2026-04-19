import joblib

model = joblib.load("checkpoints/random_forest_model.pkl")

print("Random Forest Model Summary")
print("===========================")
print("Number of Trees:", len(model.estimators_))
print("Max Depth:", model.max_depth)
print("Number of Features:", model.n_features_in_)

