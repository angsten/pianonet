import numpy as np


class NoteSampleGenerator(object):
    """
    Class representing a generator of NoteArray input and target segments that can be used for training.
    """

    def __init__(self, note_arrays_list, sample_length_in_timesteps, batch_size, random_seed=0):
        self.note_arrays_list = note_arrays_list
        self.sample_length_in_timesteps = sample_length_in_timesteps
        self.batch_size = batch_size

        self.sample_indices_list = self.get_sample_indices_list()
        self.random_seed = random_seed

        self.note_array_valid_sample_lengths_in_timesteps = np.array(
            [(note_array.get_length_in_timesteps() - self.sample_length_in_timesteps) for note_array in
             self.note_arrays_list])

        self.total_valid_sample_timesteps = np.sum(self.note_array_valid_sample_lengths_in_timesteps)

        self.cumulative_timestep_lengths_array = np.cumsum(self.note_array_valid_sample_lengths_in_timesteps)

        np.random.seed(self.random_seed)
        self.random_indices_list = np.random.choice(a=self.total_valid_sample_timesteps,
                                                    size=self.total_valid_sample_timesteps,
                                                    replace=False)

    def __iter__(self):
        # sample an index, say ind = 5000
        #
        # need to find which note array this corresponds to - find first point when larger than cumsum value
        #
        # np.random.seed(0)

        while (True):
            yield None
