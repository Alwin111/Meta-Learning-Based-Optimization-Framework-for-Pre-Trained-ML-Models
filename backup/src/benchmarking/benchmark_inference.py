import time

def benchmark_model(session, input_data, runs=10, warmup=3):

    input_name = session.get_inputs()[0].name

    # warmup runs
    for _ in range(warmup):
        session.run(None, {input_name: input_data})

    latencies = []

    for _ in range(runs):

        start = time.time()

        session.run(None, {input_name: input_data})

        end = time.time()

        latencies.append(end - start)

    return sum(latencies) / len(latencies)
