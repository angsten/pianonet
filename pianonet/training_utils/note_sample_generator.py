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

        self.master_note_array = master_note_array
        self.num_notes_in_model_input = num_notes_in_model_input
        self.num_predicted_notes_in_sample = num_predicted_notes_in_sample
        self.batch_size = batch_size
        self.random_seed = random_seed

        self.total_notes = self.master_note_array.note_array.get_length_in_notes()

        prediction_start_indices = np.arange(start=0,
                                             stop=self.total_notes,
                                             step=self.num_predicted_notes_in_sample)

        print("Prediction start indices: ", str(prediction_start_indices))

        np.random.seed(self.random_seed)

        self.randomized_prediction_start_indices = prediction_start_indices

        # NOW SHUFFLE IT!!!!!!!!!!!!!!!!!!!!!!!!

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

    # def __iter__(self):
    #     return iter(self.batch_generator())

    def get_batch_generator(self):

        def batch_generator():
            counter = 0

            while (True):

                inputs = []
                targets = []

                for i in range(self.batch_size):
                    prediction_start_index = self.randomized_prediction_start_indices[counter]

                    print(prediction_start_index)

                    input_index_range = self.get_input_sample_index_range(
                        prediction_start_index=prediction_start_index)

                    print("Input index range:" + str(input_index_range))

                    target_index_range = (input_index_range[0] + 1, input_index_range[1] + 1)

                    bounded_input_index_range = (max(input_index_range[0], 0),
                                                 min(input_index_range[1], self.total_notes - 1))

                    bounded_target_index_range = (max(target_index_range[0], 0),
                                                  min(target_index_range[1], self.total_notes - 1))

                    input = self.master_note_array.note_array.array[
                            bounded_input_index_range[0]:bounded_input_index_range[1]]
                    target = self.master_note_array.note_array.array[
                             bounded_target_index_range[0]:bounded_target_index_range[1]]

                    if input_index_range[0] < 0:
                        pad_count = abs(input_index_range[0])
                        input = np.pad(array=input, pad_width=(pad_count, 0), mode='constant').astype('bool')
                    elif input_index_range[1] >= self.total_notes:
                        pad_count = input_index_range[1] - (self.total_notes - 1)
                        input = np.pad(array=input, pad_width=(0, pad_count), mode='constant').astype('bool')

                    if target_index_range[0] < 0:
                        pad_count = abs(target_index_range[0])
                        target = np.pad(array=target, pad_width=(pad_count, 0), mode='constant').astype('bool')
                    elif target_index_range[1] >= self.total_notes:
                        pad_count = target_index_range[1] - (self.total_notes - 1)
                        target = np.pad(array=target, pad_width=(0, pad_count), mode='constant').astype('bool')

                    inputs.append(input)
                    targets.append(target)

                    counter += 1

                    if counter >= len(self.randomized_prediction_start_indices):
                        counter = 0

                yield (np.array(inputs), np.array(targets))

        return batch_generator()
