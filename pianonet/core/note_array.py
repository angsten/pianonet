import numpy as np


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

    def __init__(self, pianoroll=None, flat_array=None, note_array_transformer=None):
        """
        pianoroll: Instance of Pianoroll class used to populate the notearray's array
        flat_array: Optionally can initialize from a 1D array of note states. **This 1D array is assumed to already
                    be cropped and downsampled at the specified parameters given to this constructor**
        note_array_transformer: NoteArrayTransformer instance for converting a pianoroll or flat array into a NoteArray
        """

        self.note_array_transformer = note_array_transformer

        pianoroll_is_defined = (pianoroll != None)
        flat_array_is_defined = isinstance(flat_array, np.ndarray)

        if pianoroll_is_defined and flat_array_is_defined:
            raise Exception("Cannot use both a pianoroll and flat_array initializer. Choose one.")

        elif pianoroll_is_defined:
            self.array = self.note_array_transformer.get_flat_array_from_pianoroll(pianoroll=pianoroll)

        elif flat_array_is_defined:
            self.note_array_transformer.validate_flat_array(flat_array)
            self.array = flat_array.copy()

        else:
            raise Exception("Neither a pianoroll nor a flat_array initializer has been provided.")

    def get_pianoroll(self):
        """
        Recover the original pianoroll as high of fidelity as possible given the initial down-sampling and cropping.
        A Pianoroll instance is returned.
        """

        return self.note_array_transformer.get_pianoroll_from_flat_array(flat_array=self.array)

    def get_length_in_notes(self):
        """
        Returns as an integer the length of the stored 1D array
        """

        return self.array.shape[0]

    def get_length_in_timesteps(self):
        """
        Returns as an integer the length of the note array in timesteps
        """

        return (self.get_length_in_notes() // self.note_array_transformer.num_keys)

    def get_values_in_range(self, start_index, end_index, use_zero_padding_for_out_of_bounds=False):
        """
        start_index: Start index of desired note array values
        end_index: End index (non-inclusive) of desired note array values
        use_zero_padding_for_out_of_bounds: If true, zeros are returned for those indices in the range that are
                                            out of bounds.
        """

        pad_count_at_start = 0
        pad_count_at_end = 0

        bounded_start_index = max(start_index, 0)
        bounded_end_index = min(end_index, self.get_length_in_notes())

        values = self.array[bounded_start_index:bounded_end_index]

        if start_index < 0:
            pad_count_at_start = abs(start_index)

        if end_index > self.get_length_in_notes():
            pad_count_at_end = end_index - self.get_length_in_notes()

        if (pad_count_at_start + pad_count_at_end) > 0:
            values = np.pad(array=values,
                            pad_width=(pad_count_at_start, pad_count_at_end),
                            mode='constant').astype('bool')

        return values
