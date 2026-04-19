import time

def run_inference(model, X):
    start = time.time()
    preds = model.predict(X)
    latency = time.time() - start
    return preds, latency
