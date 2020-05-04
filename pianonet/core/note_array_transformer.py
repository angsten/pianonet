import numpy as np

from pianonet.core.note_array import NoteArray
from pianonet.core.pianoroll import Pianoroll


class NoteArrayTransformer(object):
    """
    Class for transforming a Pianoroll instance to a NoteArray instance using cropping and down-sampling. Storing
    the transformation in an object is convenient when you have a session that is constantly using the same cropping
    and downsampling parameters.
    """

    def __init__(self, min_key_index=0, num_keys=128, resolution=1.0):
        """
        min_key_index: array index of the lowest piano key to not crop
        num_keys: number of piano keys in pianoroll to keep. min_key_index + num_keys - 1 gives the index of the highest
                  included key.
        resolution: Float between 0 and 1 controlling how much to downsample the original pianoroll array.
                    resolution=0.5 means every other note is sampled. Must be 1.0 if a flat_array is given
        """

        self.min_key_index = min_key_index
        self.num_keys = num_keys
        self.resolution = resolution

        self.max_key_index = self.min_key_index + self.num_keys

        self.validate_attributes()

    def validate_attributes(self):
        """
        Makes sure parameters are valid, raises an exception if not.
        """

        if (self.min_key_index + self.num_keys) > 128:
            raise Exception("(min_key_index + num_keys) = " + str(
                self.min_key_index + self.num_keys) + ", which is greater than 128")

        if self.resolution > 1.0:
            raise Exception("Provided resolution of " + str(self.resolution) + " is greater than 1.0.")

    def get_flat_array_from_pianoroll(self, pianoroll):
        """
        pianoroll: Pianoroll instance to generate NoteArray from

        Generates 1D flat array of boolean key states from the Pianoroll instance pianoroll.
        """

        downsampled_pianoroll = pianoroll.get_stretched(stretch_fraction=self.resolution)

        cropped_pianoroll = downsampled_pianoroll.array[:, self.min_key_index:self.max_key_index]

        flat_array = cropped_pianoroll.flatten()

        return flat_array

    def get_pianoroll_from_flat_array(self, flat_array):
        """
        flat_array: 1D array of boolean note states.

        Return a Pianoroll instance by reversing the cropping and downsampling transformation previously applied.
        """

        self.validate_flat_array(flat_array)

        cropped_unflattened_pianoroll_array = flat_array.reshape(flat_array.shape[0] // self.num_keys, self.num_keys)

        unflattened_pianoroll_array = np.zeros((flat_array.shape[0] // self.num_keys, 128)).astype('bool')

        unflattened_pianoroll_array[:,
        self.min_key_index:self.min_key_index + self.num_keys] = cropped_unflattened_pianoroll_array

        return Pianoroll(unflattened_pianoroll_array).get_stretched(stretch_fraction=(1.0 / self.resolution))

    def validate_flat_array(self, flat_array):
        """
        flat_array: 1D array of boolean key states.

        Raises exception if flat_array size is not a multiple of num_keys.
        """

        flat_array_size_is_multiple_of_num_keys = ((flat_array.shape[0] % self.num_keys) == 0)

        if not flat_array_size_is_multiple_of_num_keys:
            raise Exception("flat_array contains a timestep with a partial key state. This is not allowed.")

    def get_note_array(self, pianoroll=None, flat_array=None):
        """
        pianoroll: Pianoroll instance to generate NoteArray from
        flat_array: Only if pianoroll is None, the 1D array of bools for populating the NoteArray data
        Generates new NoteArray instance from a Pianoroll instance pianoroll or a 1D array of bools, flat_array.
        """

        return NoteArray(pianoroll=pianoroll,
                         flat_array=flat_array,
                         min_key_index=self.min_key_index,
                         num_keys=self.num_keys,
                         resolution=self.resolution)
