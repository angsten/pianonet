import numpy as np


class NoteSampleGenerator(object):
    """
    Class representing a generator of NoteArray input and target segments that can be used for training.
    """

    def __init__(self,
                 master_note_array,
                 num_notes_in_model_input,
                 num_predicted_notes_in_sample,
                 batch_size,
                 random_seed=0):
        """

        master_note_array: MasterNoteArray instance containing the array of notes that will be used in training
        num_notes_in_model_input: The size of the expected input for the 1D Conv Net
        num_predicted_notes_in_sample: How many predictions will be generated per sample (the model's 'headway' for
                                       sliding forward)
        batch_size: How many pairs of input and target arrays to return per generator call
        random_seed: Integer for controlling randomization of the sample indices
        """

        self.master_note_array = master_note_array
        self.num_notes_in_model_input = num_notes_in_model_input
        self.num_predicted_notes_in_sample = num_predicted_notes_in_sample
        self.batch_size = batch_size
        self.random_seed = random_seed

        self.total_notes = self.master_note_array.note_array.get_length_in_notes()

        prediction_start_indices = np.arange(start=0,
                                             stop=self.total_notes,
                                             step=self.num_predicted_notes_in_sample)

        np.random.seed(self.random_seed)
        # NOW SHUFFLE IT!!!!!!!!!!!!!!!!!!!!!!!!
        self.randomized_prediction_start_indices = prediction_start_indices ##Add np.random.choice later


    def get_input_sample_index_range(self, prediction_start_index):
        """
        Get the start and end indices of the sample input using the following scheme:

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

    def get_batch_generator(self):
        """
        Generate a batch of samples.
        """

        def batch_generator():
            start_indices_index = 0

            while (True):

                inputs = []
                targets = []

                for i in range(self.batch_size):
                    prediction_start_index = self.randomized_prediction_start_indices[start_indices_index]

                    print("Prediction start index: ", str(prediction_start_index))

                    input_index_range = self.get_input_sample_index_range(prediction_start_index=prediction_start_index)

                    print("Input index range:" + str(input_index_range))

                    input = self.master_note_array.note_array.get_values_in_range(
                        start_index=input_index_range[0],
                        end_index=input_index_range[1],
                        use_zero_padding_for_out_of_bounds=True)

                    target = self.master_note_array.note_array.get_values_in_range(
                        start_index=input_index_range[0] + 1,
                        end_index=input_index_range[1] + 1,
                        use_zero_padding_for_out_of_bounds=True)

                    inputs.append(input)
                    targets.append(target)

                    start_indices_index += 1

                    if start_indices_index >= len(self.randomized_prediction_start_indices):
                        start_indices_index = 0

                yield (np.array(inputs), np.array(targets))

        return batch_generator()
