import joblib

model = joblib.load("optimized_model.pkl")

print(type(model))
