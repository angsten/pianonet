from pianonet.core.pianoroll import Pianoroll
from pianonet.core.note_array import NoteArray
from pianonet.core.note_array_creator import NoteArrayCreator

from pianonet.core.midi_tools import get_midi_file_paths_list


class NoteStreamCreator(object):
    """
    This class will iterate through a folder containing midi files and concatenate their data into a single NoteArray
    instance that can be used for training on. Padding is added to the beginning of each midi file before concatenating
    to ensure the model is note predicting the next song from the previous song's notes. Having one large note array
    to sample from ensures that longer songs are sampled more than shorter ones, producing a more representative model.
    """

    def __init__(self,
                 path_to_directory_of_midi_files,
                 note_array_creator,
                 augmentations_per_midi_file=1,
                 stretch_range=None,
                 left_padding=0):

        self.path_to_directory_of_midi_files = path_to_directory_of_midi_files
        self.note_array_creator = note_array_creator
        self.augmentations_per_midi_file = augmentations_per_midi_file
        self.stretch_range = stretch_range
        self.left_padding = left_padding


