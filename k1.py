import joblib

model = joblib.load("random_forest_model.pkl")

print(type(model))
print("Depth:", model.get_depth())
print("Nodes:", model.tree_.node_count)
