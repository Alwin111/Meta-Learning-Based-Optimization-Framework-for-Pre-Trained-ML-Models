import time
import numpy as np

def benchmark_model(session, input_data, runs=50):

    input_name = session.get_inputs()[0].name

    times = []

    for _ in range(runs):

        start = time.time()

        session.run(None, {input_name: input_data})

        end = time.time()

        times.append(end - start)

    avg_latency = np.mean(times)

    return avg_latency
