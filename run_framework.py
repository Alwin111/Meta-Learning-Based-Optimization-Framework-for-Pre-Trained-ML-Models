import sys
from run_experiment import main
from src.meta_learning.optimization_selector import recommend_optimization
from src.utils.result_logger import log_meta_result

# choose model
model_name = "mobilenet"
model_type = "cnn"

# get recommended optimization
recommended = recommend_optimization(model_name)

print("Recommended model configuration:", recommended)

# run experiment
sys.argv = ["run_experiment.py", "configs/random_forest.yaml"]

metrics = main()

# metrics expected from experiment
accuracy = metrics["accuracy"]
latency = metrics["avg_inference_time_sec"]
model_size = metrics["model_size_mb"]

# log into meta dataset
log_meta_result(
    model_name=model_name,
    model_type=model_type,
    optimization=recommended,
    accuracy=accuracy,
    latency=latency,
    model_size=model_size
)
