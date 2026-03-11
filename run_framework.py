import sys
from run_experiment import main
from src.meta_learning.optimization_selector import recommend_optimization

recommended = recommend_optimization("mobilenet")
print("Recommended model configuration:", recommended)

# pass config file to run_experiment
sys.argv = ["run_experiment.py", "configs/random_forest.yaml"]

main()
