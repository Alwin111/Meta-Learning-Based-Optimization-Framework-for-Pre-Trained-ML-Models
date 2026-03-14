import pandas as pd


def recommend_optimization(model_name):

    # read experiment results
    df = pd.read_csv("experiments/results.csv")

    # filter rows containing the model name
    model_df = df[df["model"].str.contains(model_name)]

    if model_df.empty:
        return "baseline"

    # choose row with lowest inference time
    best_row = model_df.loc[model_df["avg_inference_time_sec"].idxmin()]

    return best_row["model"]
