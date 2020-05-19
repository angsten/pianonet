import numpy as np

from pianonet.core.misc_tools import get_hash_string_of_numpy_array
from pianonet.core.misc_tools import save_dictionary_to_json_file, load_dictionary_from_json_file


class NoteSampleGenerator(object):
    """
    Class representing a generator of NoteArray input and target segments that can be used for training a 1D conv net.

    The 'prediction start indices' are the positions in the flat array of note states that give the starts of the series
    of note states to be predicted in each sample. These start positions are each num_predicted_notes_in_sample distance
    apart. Each sample is given relative to the prediction start index, as diagrammed in the comment of the
    get_input_sample_index_range method. A constant random_seed means the same order of randomized prediction start
    indices will be generated, so the samples will look identical if the master note array is kept constant.

    The sampling method used below guarantees that each note in the master note array will be predicted exactly once
    in each training epoch, if the epoch runs for get_total_samples_count() iterations. Note: Zero padding is added at
    the boundaries to ensure all notes are sampled, adding a very small number of additional 0-state input notes
    at the beginning and 0-state predicted notes at the end.
    """

    def __init__(self,
                 master_note_array,
                 num_notes_in_model_input,
                 num_predicted_notes_in_sample,
                 batch_size,
                 random_seed=0):
        """
        master_note_array: MasterNoteArray instance containing the array of note states that will be used in training
        num_notes_in_model_input: The size of the expected input for the 1D convnet
        num_predicted_notes_in_sample: Predicted notes given in each sample (the model's 'headway' for sliding forward)
        batch_size: How many pairs of input and target arrays to return per generator call
        random_seed: Integer for controlling randomization of the sampled start indices
        """

        self.master_note_array = master_note_array
        self.num_notes_in_model_input = num_notes_in_model_input
        self.num_predicted_notes_in_sample = num_predicted_notes_in_sample
        self.batch_size = batch_size

        self.prediction_start_indices_index = 0
        self.full_runs_through_data_count = 0

        # Each sample will be generated relative to these indices, which track where the series of predicted notes begin
        self.randomized_prediction_start_indices = np.arange(start=0,
                                                             stop=self.master_note_array.get_length_in_notes(),
                                                             step=self.num_predicted_notes_in_sample)

        np.random.seed(random_seed)
        np.random.shuffle(self.randomized_prediction_start_indices)

    def __iter__(self):
        """
        Allows this class to serve as an iterable object.
        """

        return self

    def get_total_samples_count(self):
        """
        Returns total number of unique samples this generator can output before looping back through the data.
        """

        return len(self.randomized_prediction_start_indices)

    def get_total_batches_count(self):
        """
        Returns total number of unique batches of samples this generator can output before looping back through
        the data.
        """

        return self.get_total_samples_count() // self.batch_size

    def get_fraction_data_seen(self):
        """
        Returns the fraction of data already generated. For example, if all samples have been generated once as well as
        30% of the second pass through, this method will return 1.3.
        """

        return self.full_runs_through_data_count + self.prediction_start_indices_index / self.get_total_samples_count()

    def set_prediction_start_indices_index(self, prediction_start_indices_index):
        """
        prediction_start_indices_index: The new prediction_start_indices_index to set the index to.
        """

        self.prediction_start_indices_index = prediction_start_indices_index

    def get_then_update_prediction_start_index(self):
        """
        Return the current position tracking the location of the next series of predictions returned from the master
        note array data. This method then increments the current index to be the next valid start position.
        """

        prediction_start_index = self.randomized_prediction_start_indices[self.prediction_start_indices_index]

        self.prediction_start_indices_index += 1

        if self.prediction_start_indices_index >= self.get_total_samples_count():
            self.prediction_start_indices_index = 0
            self.full_runs_through_data_count += 1

        return prediction_start_index

    def get_input_sample_index_range(self, prediction_start_index):
        """
        Get the start and end indices of the sample input array using the following scheme:

            num_notes_in_model_input          num_predicted_notes_in_sample - 1
        | - - - - - - - - - - - - - - | | - - - - - - - - - - - - - - - - - - - - - |

        0 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 1 0
        ^                               ^                                           ^
        |                               |                                           |
        start_index                     prediction_start_index                      end_index
        """

        start_index = prediction_start_index - self.num_notes_in_model_input

        end_index = prediction_start_index + (self.num_predicted_notes_in_sample - 1)

        return (start_index, end_index)

    def __next__(self):
        """
        Generate the next batch of samples taken from the master note array data based on what the current prediction
        start index is.
        """

        inputs = []
        targets = []

        for i in range(self.batch_size):
            prediction_start_index = self.get_then_update_prediction_start_index()

            input_index_range = self.get_input_sample_index_range(prediction_start_index=prediction_start_index)

            input = self.master_note_array.get_values_in_range(
                start_index=input_index_range[0],
                end_index=input_index_range[1],
                use_zero_padding_for_out_of_bounds=True)

            target = self.master_note_array.get_values_in_range(
                start_index=(input_index_range[0] + 1) + (self.num_notes_in_model_input - 1),
                end_index=input_index_range[1] + 1,
                use_zero_padding_for_out_of_bounds=True)

            inputs.append(input.reshape((-1, 1)))
            targets.append(target.reshape((-1, 1)))

        return (np.array(inputs), np.array(targets))

    def get_identifier_hash_string(self):
        """
        Returns a string that serves as a unique identifier of the generator. If this hash is the same, the data
        and how it's indexed are exactly the same. Note, this hash does not include the state of the generator,
        such as where the current prediction index is located.
        """

        return self.master_note_array.get_hash_string() + " " + get_hash_string_of_numpy_array(
            self.randomized_prediction_start_indices)

    def get_summary_string(self):
        """
        Returns summary string that is useful for quickly comparing whether two sample generators are the same.
        """

        summary_string = "- " * 50
        summary_string += "\nGenerator identifier hash string: " + self.get_identifier_hash_string()

        summary_string += "\n\nTotal notes in generator: " + '{:,}'.format(self.master_note_array.get_length_in_notes())
        summary_string += "\nTotal samples in generator: " + '{:,}'.format(self.get_total_samples_count())
        summary_string += "\nBatch size: " + str(self.batch_size)
        summary_string += "\nNumber of time steps of model input in each sample: " + '{:,}'.format(
            self.num_notes_in_model_input // self.master_note_array.note_array_transformer.num_keys)
        summary_string += "\nNumber of predicted time steps in each sample: " + '{:,}'.format(
            self.num_predicted_notes_in_sample // self.master_note_array.note_array_transformer.num_keys)

        summary_string += "\n\nPercent of data seen: " + str(round(self.get_fraction_data_seen() * 100, 2)) + "%"
        summary_string += "\nPrediction start indices index: " + '{:,}'.format(self.prediction_start_indices_index)

        summary_string += "\n" + "- " * 50

        return summary_string

    def get_state_dictionary(self):
        """
        Returns dictionary specifying state of this generator. The state is basically where in the dataset this
        generator currently points, making sure progress is tracked. The state is NOT the data or the prediction
        indices.
        """

        state = {}

        state['prediction_start_indices_index'] = self.prediction_start_indices_index
        state['full_runs_through_data_count'] = self.full_runs_through_data_count

        return state

    def set_state(self, state_dictionary):
        """
        Set this generator's state from a given state_dictionary.

        state_dictionary: A dictionary containing the state of a generator (as defined in get_state_dictionary).
        """

        self.prediction_start_indices_index = state_dictionary['prediction_start_indices_index']
        self.full_runs_through_data_count = state_dictionary['full_runs_through_data_count']

    def save_state(self, file_path):
        """
        Save generators state to file.
        """

        save_dictionary_to_json_file(dictionary=self.get_state_dictionary(), json_file_path=file_path)

    def load_state(self, file_path):
        """
        Load a previous generator's state saved to file into this generator.
        """

        self.set_state(state_dictionary=load_dictionary_from_json_file(json_file_path=file_path))
