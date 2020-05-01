import numpy as np


class NoteSampleGenerator(object):
    """
    Class representing a generator of NoteArray input and target segments that can be used for training.
    """

    def __init__(self, note_arrays_list, sample_length_in_timesteps, batch_size, random_seed=0):
        self.note_arrays_list = note_arrays_list
        self.sample_length_in_timesteps = sample_length_in_timesteps
        self.batch_size = batch_size
        self.random_seed = random_seed

        self.num_note_arrays = len(self.note_arrays_list)

        valid_sample_points_table = np.zeros((100, 2), dtype='int')

        valid_sample_points_list = []

        last_timestep_index = 0
        for note_array_index in range(self.num_note_arrays):
            note_array = self.note_arrays_list[note_array_index]

            print("Note array length: " + str(note_array.get_length_in_timesteps()))

            valid_sample_timesteps = np.arange(
                start=0,
                stop=note_array.get_length_in_timesteps(),   ####if not modded correctly, could result in too large of index!
                step=self.sample_length_in_timesteps,
                dtype='int',
            )

            for sample_timestep in valid_sample_timesteps:
                valid_sample_points_list.append((note_array_index, sample_timestep))


        print(valid_sample_points_list)

        # np.random.seed(self.random_seed)

        # NOW SHUFFLE IT!!!!!!!!!!!!!!!!!!!!!!!!

    def __iter__(self):
        # sample an index, say ind = 5000
        #
        # need to find which note array this corresponds to - find first point when larger than cumsum value
        #

        counter = 0
        while (True):

            inputs = []
            targets = []

            for i in range(self.batch_size):

                yield np.array(inputs, targets)

                counter += 1

    def save(self):
        pass
