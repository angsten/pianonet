import time

from tensorflow.keras.callbacks import Callback


class ExecuteEveryNBatchesAndEpochCallback(Callback):
    """
    Will run a method method_to_run every run_frequency_in_batches batches, and, if provided, method_to_run_on_epoch_end
    at the end of each epoch.
    """

    def __init__(self, run_frequency_in_batches, method_to_run, method_to_run_on_epoch_end=None):
        """
        run_frequency_in_batches: Integer specifying how many batches must pass before method_to_run is called
        method_to_run: Method to call each time run_frequency_in_batches have completed
        method_to_run_on_epoch_end: Method to call upon the end of each epoch

        Note: Both method_to_run and method_to_run_on_epoch_end methods must accept a batch/epoch integer and a
        logs dictionary as parameters.
        """

        self.batches_seen = 0
        self.run_frequency_in_batches = run_frequency_in_batches
        self.method_to_run = method_to_run
        self.method_to_run_on_epoch_end = method_to_run_on_epoch_end

    def on_train_batch_end(self, batch, logs={}):
        """
        Method called at the end of each batch.

        batch: Integer specifying the current batch count within the current epoch
        logs: Dictionary of logs passed by tensorflow carrying batch information (the loss on the batch)
        """

        self.batches_seen += 1
        if (self.batches_seen % self.run_frequency_in_batches == 0):
            self.method_to_run(batch, logs)

    def on_epoch_begin(self, epoch=None, logs=None):
        """
        Method called at the beginning of each epoch. Here we set the timing start data.

        epoch: Integer specifying the current epoch count within the current training session
        logs: Dictionary of logs passed by tensorflow carrying epoch information (such as the average loss)
        """

        self.start_wall_time = time.time()
        self.start_cpu_time = time.clock()

    def on_epoch_end(self, epoch=None, logs=None):
        """
        Method called at the end of each epoch.

        epoch: Integer specifying the current epoch count within the current training session
        logs: Dictionary of logs passed by tensorflow carrying epoch information (such as the average loss)
        """

        if self.method_to_run_on_epoch_end:
            logs['start_wall_time'] = self.start_wall_time
            logs['start_cpu_time'] = self.start_cpu_time
            self.method_to_run_on_epoch_end(epoch, logs)
