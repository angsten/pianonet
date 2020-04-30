import numpy as np

from pianonet.core.midi_tools import get_midi_file_paths_list
from pianonet.core.pianoroll import Pianoroll

from pianonet.core.misc_tools import get_noisily_spaced_floats


class MasterNoteArraysListCreator(object):
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
                 start_padding_timesteps=0,
                 end_padding_timesteps=0,
                 ):
        """
        path_to_directory_of_midi_files: String path to directory of midi files one level deep in file tree
        note_array_creator: NoteArrayCreator instance specifying how to crop and downsample a pianoroll into a NoteArray
        num_augmentations_per_midi_file: How many stretched pianorolls to create from each midi fil
        stretch_range: A tupe of two floats in range (0.0, infinity) specifying valid random range for stretch fractions
        start_padding_timesteps: How much padding in timesteps to add between two concatentated pianorolls
        end_padding_timesteps: How much padding in timesteps to add between two concatentated pianorolls
        """
        self.path_to_directory_of_midi_files = path_to_directory_of_midi_files
        self.note_array_creator = note_array_creator
        self.num_augmentations_per_midi_file = num_augmentations_per_midi_file
        self.stretch_range = stretch_range if (stretch_range != None) else [1.0, 1.0]
        self.start_padding_timesteps = start_padding_timesteps
        self.end_padding_timesteps = end_padding_timesteps

    def get_note_arrays_list(self):
        """
        Creates the master note array using the following steps:

        1. Get list of midi files in directory
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

        [print(m) for m in midi_file_paths_list]

        note_arrays_list = []

        for midi_file_path in midi_file_paths_list:

            print("Using file at " + midi_file_path)

            pianoroll = Pianoroll(midi_file_path)

            pianoroll = pianoroll[0:100]  #####remove!!

            stretch_fractions = get_noisily_spaced_floats(start=self.stretch_range[0],
                                                          end=self.stretch_range[1],
                                                          num_points=self.num_augmentations_per_midi_file)

            print("Stretch_fractions array:", str(stretch_fractions))

            for i in range(self.num_augmentations_per_midi_file):
                stretch_fraction = stretch_fractions[i]

                stretched_pianoroll = pianoroll.get_stretched(stretch_fraction=stretch_fraction)

                stretched_pianoroll.add_zero_padding(left_padding_timesteps=self.start_padding_timesteps,
                                                     right_padding_timesteps=self.end_padding_timesteps)

                note_array = self.note_array_creator.get_instance(pianoroll=stretched_pianoroll)

                note_arrays_list.append(note_array)

        return note_arrays_list
