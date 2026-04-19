import matplotlib.pyplot as plt

def plot_latency(latency):
    labels = ["Baseline", "Optimized"]
    values = [latency, latency * 0.7]  # dummy improvement

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_ylabel("Latency")

    return fig
