from tensorflow.keras.callbacks import Callback

class ExecuteEveryNBatchesCallback(Callback):
    """
    Will run a method method_to_run every run_frequency_in_batches batches. The method is NOT run
    before batch zero.
    """

    def __init__(self, run_frequency_in_batches, method_to_run):
        self.batches_seen = 0
        self.run_frequency_in_batches = run_frequency_in_batches
        self.method_to_run = method_to_run

    def on_batch_end(self, batch, logs={}):
        self.batches_seen += 1
        if (self.batches_seen != 0) and (self.batches_seen % self.run_frequency_in_batches == 0):
            self.method_to_run(batch, logs)