import numpy as np

from pianonet.core.pianoroll import Pianoroll


class NoteArray(object):
    """
    A NoteArray is a 1D stream of piano note states derived from flattening a pianoroll. The notearray is useful
    for training a 1D convolutional neural net. The parent pianoroll can be set to lower resolution before flattening.
    Also, most keys in pianorolls are always or nearly always zero, usually the highest and lowest keys. This is why
    cropping high and low keys of the input pianoroll is supported.

    Example array:

    timestep = 0                 timestep = 1
     A  A# B  C  C# D  D# E  ... A  A# B  C  C# D  D# E  ...
    [0, 0, 0, 1, 0, 0, 0, 0, ... 1, 0, 0, 1, 0, 0, 0, 0, ...]
    """

    def __init__(self, pianoroll=None, flat_array=None, min_key_index=0, num_keys=128, resolution=1.0):
        """
        pianoroll: Instance of Pianoroll class used to populate the notearray's array
        flat_array: Optionally can initialize from a 1D array of note states.
        min_key_index: array index of the lowest piano key to not crop
        num_keys: number of piano keys in pianoroll to keep. min_key_index + num_keys - 1 gives the index of the highest
                  included key.
        resolution: Float between 0 and 1 controlling how much to downsample the original pianoroll array.
                    resolution=0.5 means every other note is sampled. Must be 1.0 if a flat_array is given
        """

        if (pianoroll != None) and (flat_array != None):
            raise Exception("Cannot use both a pianoroll and flat_array initializer. Choose one.")

        if flat_array != None:
            if (flat_array.shape[0] % num_keys) != 0:
                raise Exception("flat_array contains a timestep with a partial key state. This is not allowed.")

            num_time_steps = (flat_array.shape[0] // num_keys)
            pianoroll = Pianoroll(flat_array.reshape((num_time_steps, num_keys)))

        pianoroll = pianoroll.get_copy()

        self.min_key_index = min_key_index
        self.num_keys = num_keys
        self.resolution = resolution

        if self.resolution != 1.0:
            pianoroll.stretch(stretch_fraction=resolution)

        self.time_steps = pianoroll.array.shape[0]

        cropped_pianoroll = pianoroll.array[:, min_key_index:(min_key_index + num_keys)]

        self.array = cropped_pianoroll.flatten()

    def get_pianoroll(self):
        """
        Recover the original pianoroll as high of fidelity as possible given the initial down-sampling and cropping.
        A Pianoroll instance is returned.
        """

        cropped_unflattened_pianoroll_array = np.reshape(self.array, (self.time_steps, self.num_keys))

        unflattened_pianoroll_array = np.zeros((self.time_steps, 128)).astype('bool')

        unflattened_pianoroll_array[:,
        self.min_key_index:self.min_key_index + self.num_keys] = cropped_unflattened_pianoroll_array

        pianoroll_low_resolution = Pianoroll(unflattened_pianoroll_array)

        if self.resolution == 1.0:
            return pianoroll_low_resolution
        else:
            return pianoroll_low_resolution.get_stretched(stretch_fraction=(1.0 / self.resolution))
