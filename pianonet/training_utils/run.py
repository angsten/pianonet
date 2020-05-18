import os
import json
from keras.models import Model, load_model
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

        initial_json_path = os.path.join(path, 'run_description.json')

        self.load_state_from_json(json_path=initial_json_path)

    def load_state_from_json(self, json_path):
        with open(json_path, 'rb') as json_file:
            self.run_description = json.load(json_file)

    def set_model(self):
        model_description = self.run_description['model_description']

        if model_description['initial_model_path'] != "":
            self.model = load_model(model_description['initial_model_path'])

    def run(self):
        data_description = self.run_description['data_description']
        training_master_note_array = MasterNoteArray(file_path=data_description['training_master_note_array_path'])
        validation_master_note_array = MasterNoteArray(file_path=data_description['validation_master_note_array_path'])
        note_array_transformer = training_master_note_array.note_array_transformer

