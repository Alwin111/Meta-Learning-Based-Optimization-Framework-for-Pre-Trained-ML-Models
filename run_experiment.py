from src.models.random_forest import train_random_forest

model_name = config["model"]["name"]

if model_name == "random_forest":
    train_random_forest(config)
else:
    raise ValueError("Unsupported model")
