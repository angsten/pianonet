from pianonet.core.note_array import NoteArray


class NoteArrayCreator(object):
    """
    Class for generating NoteArray class instances using fixed cropping and downsampling parameters. This
    is convenient when you have a session that is constantly using the same cropping and downsampling parameters.
    """

    def __init__(self, min_key_index=0, num_keys=128, resolution=1.0):
        """
        For parameter definitions, see NoteArray class.
        """

        self.min_key_index = min_key_index
        self.num_keys = num_keys
        self.resolution = resolution

    def get_instance(self, pianoroll=None, flat_array=None):
        """
        For parameter definitions, see NoteArray class.

        Generates new NoteArray instance from pianoroll or flat_array.
        """
        return NoteArray(pianoroll=pianoroll,
                         flat_array=flat_array,
                         min_key_index=self.min_key_index,
                         num_keys=self.num_keys,
                         resolution=self.resolution)
