import numpy as np
import time


def run_inference(session, input_data):
    """
    Runs inference and measures latency.
    """

    input_name = session.get_inputs()[0].name

    start = time.time()
    outputs = session.run(None, {input_name: input_data})
    end = time.time()

    latency = end - start

    return outputs, latency
