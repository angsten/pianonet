from pianonet.core.pianoroll import Pianoroll
from pianonet.core.note_array import NoteArray
from pianonet.core.note_array_creator import NoteArrayCreator

from pianonet.core.midi_tools import get_midi_file_paths_list

import random


class MasterNoteArrayCreator(object):
    """
    This class will iterate through a folder containing midi files and concatenate their data into a single very large
    NoteArray instance, called the 'master note array', that can be used for efficiently training on. Padding is added
    to the beginning of each midi file before concatenating to ensure the model is note predicting the next song from
    the previous song's notes. Having one large note array to sample from ensures that longer songs are sampled more
    than shorter ones, producing a properly calibrated model.
    """

    def __init__(self,
                 path_to_directory_of_midi_files,
                 note_array_creator,
                 num_augmentations_per_midi_file=1,
                 stretch_range=None,
                 left_padding_timesteps=0,
                 ):
        self.path_to_directory_of_midi_files = path_to_directory_of_midi_files
        self.note_array_creator = note_array_creator
        self.num_augmentations_per_midi_file = num_augmentations_per_midi_file
        self.stretch_range = stretch_range
        self.left_padding_timesteps = left_padding_timesteps

    def get_instance(self):
        """
        Creates the master note array using the following steps:

        1. Get list of midi files, randomly shuffle them.
        2. For each midi file in this list:
            a. Load file as Pianoroll instance pianoroll
            b. Generate num_augmentations_per_midi_file NoteArrays using the following steps:
                i. Stretch pianoroll by random amount within prescribed range
                ii. Add left padding to pianoroll
                iii. Create cropped and down-sampled NoteArray instance from this pianoroll
            c. Add NoteArray instances to a list
        3. Concat the full list of NoteArray instances into a single master NoteArray instance
        """
        midi_file_paths_list = get_midi_file_paths_list(self.path_to_directory_of_midi_files)

        random.shuffle(midi_file_paths_list)

        # [print(m) for m in midi_file_paths_list]

        note_arrays_list = []

        for midi_file_path in midi_file_paths_list:

            pianoroll = Pianoroll(midi_file_path)

            for i in range(self.num_augmentations_per_midi_file):
                stretch_fraction = random.uniform(self.stretch_range[0], self.stretch_range[1])
                stretched_pianoroll = pianoroll.get_stretched(stretch_fraction=stretch_fraction)

                stretched_pianoroll.add_zero_padding(left_padding_timesteps=self.left_padding_timesteps)

                note_array = self.note_array_creator.get_instance(pianoroll=stretched_pianoroll)