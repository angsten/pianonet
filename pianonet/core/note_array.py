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

    def __init__(self, pianoroll, min_key_index=0, num_keys=128, resolution=1.0):
        """
        pianoroll: Instance of Pianoroll class used to populate the notearray's array
        min_key_index: array index of the lowest piano key to not crop
        num_keys: number of piano keys in pianoroll to keep. min_key_index + num_keys - 1 gives the index of the highest
                  included key.
        resolution: Float between 0 and 1 controlling how much to downsample the original pianoroll array.
                    resolution=0.5 means every other note is sampled.
        """

        pianoroll = pianoroll.copy()

        self.resolution = resolution

        if self.resolution != 1.0:
            pianoroll.stretch(stretch_fraction=resolution)

        cropped_pianoroll = pianoroll.array[:, min_key_index:(min_key_index + num_keys)]

        self.array = cropped_pianoroll.flatten()
