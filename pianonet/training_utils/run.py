import json
import os
import pickle
import subprocess

from keras.models import load_model

from pianonet.model_building.get_model_input_shape import get_model_input_shape
from pianonet.training_utils.master_note_array import MasterNoteArray


class Run(object):
    """
    Wrapper for a single model training run with set parameters denoted by a dictionary saved to file. Every
    run sits in its own directory, which at minimum must contain a run_description.json file describing
    where to find the data, how to generate or find the initial model, and how to train the model.
    """

    def __init__(self, path):
        """
        path: Directory in which the run is executed.
        """

        self.path = path
        self.run_index = 0

        initial_json_path = os.path.join(path, 'run_description.json')

        self.load_state_from_json(json_path=initial_json_path)

    def get_run_index_prepended_file_base_name(self, file_base_name):

        return str(self.run_index) + "_" + file_base_name

    def get_full_path_from_run_file_name(self, file_name):

        return os.path.join(self.path, file_name)

    def get_index_prepended_full_path_from_run_file_base_name(self, file_base_name):

        return self.get_full_path_from_run_file_name(self.get_run_index_prepended_file_base_name(file_base_name))

    def load_state_from_json(self, json_path):
        with open(json_path, 'rb') as json_file:
            self.run_description = json.load(json_file)

    def fetch_model(self):
        model_description = self.run_description['model_description']

        if 'model_initializer' in model_description:
            model_initializer = model_description['model_initializer']
            print("Initializing model using file at " + model_initializer['path'])
            print("Params used for model initialization are " + str(model_initializer['params']))

            model_output_path = self.get_index_prepended_full_path_from_run_file_base_name("initial_model.model")
            model_parameters_file_path = self.get_full_path_from_run_file_name('model_parameters')

            with open(model_parameters_file_path, 'wb') as file:
                pickle.dump(model_initializer['params'], file)

            model_creation_command = "python " + model_initializer[
                'path'] + " " + model_parameters_file_path + " " + model_output_path
            print("\nCalling model creator with command:")
            print(model_creation_command)

            subprocess.run(model_creation_command, shell=True, check=True)

            os.remove(model_parameters_file_path)

            self.model = load_model(model_output_path)

        elif model_description['model_path'] != "":
            print("Loading model at " + model_description['model_path'])

            self.model = load_model(model_description['model_path'])
        else:
            raise Exception("No method of creating or loading the model has been specified in the run description.")

        print("\nModel has been set. Model summary:\n")

        num_notes_in_model_input = get_model_input_shape(self.model)

        time_steps_receptive_field = num_notes_in_model_input / self.note_array_transformer.num_keys

        print("Number of notes in model input: " + str(num_notes_in_model_input))
        print("Time steps in receptive field: " + str(time_steps_receptive_field))
        print("Seconds in receptive field: " + str(round((time_steps_receptive_field) / 48, 2)))
        print()

        print(self.model.summary())

    def execute(self):

        print("\nBeginning run with with index " + str(self.run_index))

        data_description = self.run_description['data_description']

        training_master_note_array_path = data_description['training_master_note_array_path']
        print("Loading training master note array from " + training_master_note_array_path)
        training_master_note_array = MasterNoteArray(file_path=training_master_note_array_path)

        validation_master_note_array_path = data_description['validation_master_note_array_path']
        print("Loading validation master note array from " + validation_master_note_array_path)
        validation_master_note_array = MasterNoteArray(file_path=validation_master_note_array_path)
        self.note_array_transformer = training_master_note_array.note_array_transformer

        print()
        self.fetch_model()

