import os
import pickle
import subprocess
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, Nadam

from pianonet.core.misc_tools import save_dictionary_to_json_file, load_dictionary_from_json_file, create_directories
from pianonet.model_building.get_model_input_shape import get_model_input_shape
from pianonet.model_inspection.print_model_specifications import print_model_specifications
from pianonet.training_utils.custom_keras_callbacks import ExecuteEveryNBatchesAndEpochCallback
from pianonet.training_utils.logger import Logger
from pianonet.training_utils.master_note_array import MasterNoteArray
from pianonet.training_utils.note_sample_generator import NoteSampleGenerator


class Run(Logger):
    """
    Wrapper for a model training session with set parameters denoted by a dictionary, run_description.json. Every
    run is contained its own directory, self.path, which at minimum must contain a run_description.json file describing
    where to find the data, how to generate or find the initial model, and how to train the model. The run class
    smoothly handles re-running the training session from a checkpointed state, including saving the data generator
    state.
    """

    def __init__(self, path, mode):
        """
        path: Directory in which the run is executed and containing, at minimum, a file called run_description.json
        mode: Whether to run in 'train' mode for training or 'evaluate' mode for evaluating on the validation set.
        """

        self.path = os.path.abspath(path)
        self.mode = mode

        super().__init__(
            logger_name=__name__,
            log_file_path=self.get_full_run_path(file_name='output_{mode}.log'.format(mode=self.mode)),
            tf_logger=tf.get_logger()
        )

        for i in range(0, 3): self.log()
        self.log("*-" * 60)
        self.log("*-" * 60)
        self.log("*-*-*-*-* BEGINNING RUN AT " + self.path)
        self.log("*-" * 60)
        self.log("*-" * 60)
        for i in range(0, 2): self.log()

        initial_run_description_path = os.path.join(self.path, 'run_description.json')
        self.run_description = load_dictionary_from_json_file(json_file_path=initial_run_description_path)

        create_directories(
            parent_directory_path=self.path,
            directory_names_list=['models', 'run_descriptions', 'generator_states']
        )

        create_directories(parent_directory_path=os.path.join(self.path, 'models'), directory_names_list=['archived'])

        if os.path.exists(self.get_state_path()):
            self.log("Saved state found. Restarting the run and incrementing the run index.")
            self.load_state()
            self.state['run_index'] += 1

            previous_model_path = self.get_model_path(run_index=self.get_run_index() - 1)

            self.log("Using previous model at " + previous_model_path)
            self.run_description['model_description']['model_path'] = previous_model_path

        else:
            self.log("No previously saved state found. Starting as a new run.")
            self.state = {
                'run_index': 0,
            }

        self.log("State of run before execution is: ")
        self.log(self.state)

        self.log()
        self.log("Executing run description:")
        self.log(self.run_description)

        self.fetch_data()
        self.fetch_model()
        self.fetch_data_generators()
        self.train()

    def get_run_index(self):
        return self.state['run_index']

    def get_full_run_path(self, file_name):
        return os.path.join(self.path, file_name)

    def get_state_path(self):
        """
        Returns path within the run directory where the state data is stored.
        """

        return os.path.join(self.path, 'state.json')

    def save_state(self):
        """
        Save this instance's current state to file.
        """

        save_dictionary_to_json_file(dictionary=self.state, json_file_path=self.get_state_path())

    def load_state(self):
        """
        Loads a run's state from the state file and loads into this run's state.
        """

        self.state = load_dictionary_from_json_file(json_file_path=self.get_state_path())

    def get_model_path(self, run_index):
        return os.path.join(self.path, 'models', str(run_index) + '_trained_model')

    def get_archive_model_path(self, run_index, epoch):
        return os.path.join(self.path, 'models', 'archived',
                            'run_' + str(run_index) + '_epoch_' + str(epoch) + '_archived_model')

    def get_run_description_path(self, run_index):
        return os.path.join(self.path, 'run_descriptions', str(run_index) + '_run_description.json')

    def get_generator_state_path(self, run_index):
        return os.path.join(self.path, 'generator_states', str(run_index) + '_generator_state.gs')

    def fetch_data(self):
        """
        Based on the run_description, locate and load the training and validation data sets.
        """

        data_description = self.run_description['data_description']

        if self.mode == 'train':
            self.log()
            self.log("Loading training master note array from " + data_description['training_master_note_array_path'])
            self.training_master_note_array = MasterNoteArray(
                file_path=data_description['training_master_note_array_path'])

        self.log("Loading validation master note array from " + data_description['validation_master_note_array_path'])
        self.validation_master_note_array = MasterNoteArray(
            file_path=data_description['validation_master_note_array_path'])

        self.note_array_transformer = self.validation_master_note_array.note_array_transformer
        self.num_keys = self.note_array_transformer.num_keys

    def fetch_model(self):
        """
        Load a model into self.model, either by calling the model initializer script specified in the run description
        or by loading a model from a specified path.
        """

        model_description = self.run_description['model_description']

        self.log()
        if model_description['model_path'] != "":
            self.log("Loading model at " + model_description['model_path'])

            self.model = load_model(model_description['model_path'])

        elif 'model_initializer' in model_description:
            model_initializer = model_description['model_initializer']
            self.log("Initializing model using file at " + model_initializer['path'])
            self.log("Params used for model initialization are " + str(model_initializer['params']))

            model_output_path = os.path.join(self.path, "models", "initial_model")
            model_parameters_file_path = self.get_full_run_path('model_parameters')

            with open(model_parameters_file_path, 'wb') as file:
                pickle.dump(model_initializer['params'], file)

            model_creation_command = "python " + model_initializer[
                'path'] + " " + model_parameters_file_path + " " + model_output_path
            self.log("Calling model creator with command:")
            self.log(model_creation_command)

            try:
                subprocess.run(model_creation_command, shell=True, check=True)
            finally:
                os.remove(model_parameters_file_path)

            self.model = load_model(model_output_path)

        else:
            raise Exception("No method of creating or loading the model has been specified in the run description.")

        print_model_specifications(model=self.model, num_keys=self.num_keys, print_function=self.log)

    def fetch_data_generators(self):
        """
        Using self.run_description specifications, load in either a new training data generators or previously
        saved training data generators. Also load the validation data generator (never loaded from file).
        """

        num_notes_in_model_input = get_model_input_shape(self.model)

        if self.mode == 'train':
            training_description = self.run_description['training_description']
            training_batch_size = training_description['batch_size']
            num_predicted_notes_in_training_sample = self.num_keys * training_description[
                'num_predicted_time_steps_in_sample']

            self.training_note_sample_generator = NoteSampleGenerator(
                master_note_array=self.training_master_note_array,
                num_notes_in_model_input=num_notes_in_model_input,
                num_predicted_notes_in_sample=num_predicted_notes_in_training_sample,
                batch_size=training_batch_size,
                random_seed=0
            )

            if self.get_run_index() == 0:
                self.log("Creating a fresh training generator.")
            else:
                previous_generator_state_path = self.get_generator_state_path(run_index=self.get_run_index() - 1)
                self.log("Loading previous training generator state from path " + previous_generator_state_path)
                self.training_note_sample_generator.load_state(file_path=previous_generator_state_path)

            self.log('\n' * 2 + self.training_note_sample_generator.get_summary_string() + '\n')

        validation_description = self.run_description['validation_description']
        validation_batch_size = validation_description['batch_size']
        num_predicted_notes_in_validation_sample = self.num_keys * validation_description[
            'num_predicted_time_steps_in_sample']

        self.validation_note_sample_generator = NoteSampleGenerator(
            master_note_array=self.validation_master_note_array,
            num_notes_in_model_input=num_notes_in_model_input,
            num_predicted_notes_in_sample=num_predicted_notes_in_validation_sample,
            batch_size=validation_batch_size,
            random_seed=0
        )

        if self.mode == 'evaluate':
            self.log('\n' * 2 + self.validation_note_sample_generator.get_summary_string() + '\n')

    def save_model(self):
        """
        Saves the current model to file prepended with the current run index.
        """

        self.model.save(self.get_model_path(run_index=self.get_run_index()))

    def save_generator_state(self):
        """
        Save the training data generator's state to file prepended with the current run index.
        """

        generator_checkpoint_path = self.get_generator_state_path(run_index=self.get_run_index())
        self.training_note_sample_generator.save_state(generator_checkpoint_path)

    def archive_model_method_creator(self):
        """
        Saves an archived version of the most recent model in the models directory.
        """

        def archive_model(epoch=None, logs=None):
            self.model.save(self.get_archive_model_path(run_index=self.get_run_index(), epoch=epoch))

        return archive_model

    def checkpoint_method_creator(self):
        """
        Saves all relevant parts of the current run's training session and state to files within the run directory
        as an exact checkpoint from which a future run can be restarted without any change in the training outcome.
        """

        def checkpoint(batch=None, logs=None):
            save_dictionary_to_json_file(
                dictionary=self.run_description,
                json_file_path=self.get_run_description_path(run_index=self.get_run_index())
            )

            self.save_state()
            self.save_model()
            self.save_generator_state()

        return checkpoint

    def loss_logging_method_creator(self, steps_per_epoch, mode):
        """
        Generates a callback function for logging the loss as training progresses.

        steps_per_epoch: Integer specifying how many steps are in each training epoch
        mode: 'train' for logging training batches or 'evaluate' for validation batches
        """

        if mode == 'train':
            generator = self.training_note_sample_generator
        elif mode == 'evaluate':
            generator = self.validation_note_sample_generator

        def logging_method(batch=None, logs=None):
            if logs != {}:
                batch_string = str(batch) + '/' + str(steps_per_epoch)
                percent_data_string = str(
                    round(generator.get_fraction_data_seen() * 100, 3)) + '%'
                loss_string = str(round(logs.get('loss'), 7))

                mode_string = '        Evaluating on test set: ' if mode == 'evaluate' else ''

                log_string = mode_string + batch_string + ' ' + percent_data_string + ' ' + loss_string

                self.log(log_string)

        return logging_method

    def epoch_logging_method_creator(self):
        """
        Generates a callback function for summarizing a training epoch's statistics.
        """

        def epoch_logging_method(epoch=None, logs=None):
            percent_data_string = str(
                round(self.training_note_sample_generator.get_fraction_data_seen() * 100, 3)) + '%'
            cpu_time_taken_string = str(round(time.clock() - logs['start_cpu_time'], 1))
            wall_time_taken_string = str(round(time.time() - logs['start_wall_time'], 1))
            loss_string = str(round(logs.get('loss'), 7))
            val_loss_string = str(round(logs.get('val_loss', 0.0), 7))

            self.log()
            log_string = "Epoch {epoch} completed in {wall_time_taken_string} seconds ({cpu_time_taken_string} cpu-seconds) with {percent_data_string} data coverage - loss: {loss_string}  val_loss: {val_loss_string}"

            self.log(
                log_string.format(
                    epoch=epoch,
                    wall_time_taken_string=wall_time_taken_string,
                    cpu_time_taken_string=cpu_time_taken_string,
                    percent_data_string=percent_data_string,
                    loss_string=loss_string,
                    val_loss_string=val_loss_string,
                )
            )

            self.log()
            self.log()

        return epoch_logging_method

    def train(self):
        """
        Initiates the training loop of the run.
        """

        training_description = self.run_description['training_description']
        validation_description = self.run_description['validation_description']
        optimizer_description = training_description['optimizer_description']

        num_non_trainable_layers = training_description.get('num_non_trainable_layers', 0)

        need_to_compile_model = (self.get_run_index() == 0) or (num_non_trainable_layers != 0)

        if need_to_compile_model:
            if optimizer_description['type'] == 'Adam':
                optimizer = Adam(**optimizer_description['kwargs'])
            elif optimizer_description['type'] == 'Nadam':
                optimizer = Nadam(**optimizer_description['kwargs'])
            else:
                raise Exception("Optimizer type " + optimizer_description['type'] + " not yet supported.")

            if num_non_trainable_layers != 0:
                self.log("Freezing the first " + str(num_non_trainable_layers) + " layers.")
                for layer in self.model.layers[0:num_non_trainable_layers]:
                    layer.trainable = False

            self.log("Recompiling the model.")
            self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[])

        fraction_training_data_each_epoch = training_description['fraction_data_each_epoch']
        fraction_validation_data_each_epoch = validation_description['fraction_data_each_epoch']

        epochs = training_description['epochs']
        save_state_callback = ExecuteEveryNBatchesAndEpochCallback(
            train_run_frequency_in_batches=training_description['checkpoint_frequency_in_steps'],
            train_method_to_run=self.checkpoint_method_creator(),
            method_to_run_on_epoch_end=self.archive_model_method_creator(),
        )

        if self.mode == 'train':
            training_steps_per_epoch = int(
                fraction_training_data_each_epoch * self.training_note_sample_generator.get_total_batches_count())

            validation_steps_per_epoch = int(
                fraction_validation_data_each_epoch * self.validation_note_sample_generator.get_total_batches_count())

            self.log()
            self.log("Beginning training.")
            self.log("    Training steps per epoch: " + str(training_steps_per_epoch))
            self.log("    Validation steps per epoch: " + str(validation_steps_per_epoch))
            self.log()

            logging_callback = ExecuteEveryNBatchesAndEpochCallback(
                train_run_frequency_in_batches=1,
                test_run_frequency_in_batches=1,
                train_method_to_run=self.loss_logging_method_creator(
                    steps_per_epoch=training_steps_per_epoch,
                    mode='train',
                ),
                test_method_to_run=self.loss_logging_method_creator(
                    steps_per_epoch=validation_steps_per_epoch,
                    mode='evaluate',
                ),
                method_to_run_on_epoch_end=self.epoch_logging_method_creator(),
            )

            trainable_parameters_count = np.sum([K.count_params(c) for c in self.model.trainable_weights])
            non_trainable_parameters_count = np.sum([K.count_params(c) for c in self.model.non_trainable_weights])

            self.log(
                '    Total model parameters: {:,}'.format(trainable_parameters_count + non_trainable_parameters_count))
            self.log('    Trainable parameters: {:,}'.format(trainable_parameters_count))
            self.log('    Non-trainable parameters: {:,}'.format(non_trainable_parameters_count))
            self.log()

            self.model.fit(
                x=self.training_note_sample_generator,
                epochs=epochs,
                verbose=2,
                steps_per_epoch=training_steps_per_epoch,
                validation_data=self.validation_note_sample_generator,
                validation_steps=validation_steps_per_epoch,
                callbacks=[logging_callback, save_state_callback]
            )
        elif self.mode == 'evaluate':
            training_steps_per_epoch = 1
            validation_steps_per_epoch = self.validation_note_sample_generator.get_total_batches_count()

            self.log()
            self.log("Beginning evaluation.")
            self.log()

            logging_callback = ExecuteEveryNBatchesAndEpochCallback(
                test_run_frequency_in_batches=1,
                test_method_to_run=self.loss_logging_method_creator(
                    steps_per_epoch=validation_steps_per_epoch,
                    mode='evaluate',
                ),
            )

            self.model.evaluate(
                x=self.validation_note_sample_generator,
                verbose=2,
                steps=validation_steps_per_epoch,
                callbacks=[logging_callback]
            )
        else:
            raise Exception("Mode must be either 'train' or 'evaluate'.")
