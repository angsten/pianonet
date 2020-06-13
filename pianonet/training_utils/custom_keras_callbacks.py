import time

from tensorflow.keras.callbacks import Callback


class ExecuteEveryNBatchesAndEpochCallback(Callback):
    """
    Will run a method method_to_run every run_frequency_in_batches batches, and, if provided, method_to_run_on_epoch_end
    at the end of each epoch.
    """

    def __init__(self,
                 train_run_frequency_in_batches=0,
                 test_run_frequency_in_batches=0,
                 train_method_to_run=None,
                 test_method_to_run=None,
                 method_to_run_on_epoch_end=None,
                 ):
        """
        train_run_frequency_in_batches: Integer specifying how many training batches must pass before calling
                                        train_method_to_run
        test_run_frequency_in_batches: Integer specifying how many test batches must pass before calling
                                        test_method_to_run
        train_method_to_run: Method to call each time train_run_frequency_in_batches have completed
        test_method_to_run: Method to call each time test_run_frequency_in_batches have completed
        method_to_run_on_epoch_end: Method to call upon the end of each epoch

        Note: Both method_to_run and method_to_run_on_epoch_end methods must accept a batch/epoch integer and a
        logs dictionary as parameters.
        """

        self.train_batches_seen = 0
        self.test_batches_seen = 0
        self.train_run_frequency_in_batches = train_run_frequency_in_batches
        self.test_run_frequency_in_batches = test_run_frequency_in_batches
        self.train_method_to_run = train_method_to_run
        self.test_method_to_run = test_method_to_run
        self.method_to_run_on_epoch_end = method_to_run_on_epoch_end

    def on_train_batch_end(self, batch, logs={}):
        """
        Method called at the end of each batch.

        batch: Integer specifying the current batch count within the current epoch
        logs: Dictionary of logs passed by tensorflow carrying batch information (the loss on the batch)
        """

        if self.train_method_to_run != None:
            self.train_batches_seen += 1
            if ((self.train_batches_seen % self.train_run_frequency_in_batches) == 0):
                self.train_method_to_run(batch, logs)

    def on_test_batch_end(self, batch, logs={}):
        """
        Method called at the end of each batch.

        batch: Integer specifying the current batch count within the current epoch
        logs: Dictionary of logs passed by tensorflow carrying batch information (the loss on the batch)
        """

        if self.test_method_to_run != None:
            self.test_batches_seen += 1
            if ((self.test_batches_seen % self.test_run_frequency_in_batches) == 0):
                self.test_method_to_run(batch, logs)

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
