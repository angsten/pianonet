import numpy as np

from pianonet.core.midi_tools import get_midi_file_paths_list
from pianonet.core.misc_tools import get_noisily_spaced_floats
from pianonet.core.pianoroll import Pianoroll


class MasterNoteArray(object):
    """
    A collection of NoteArrays concatenated into one note array with metadata tracked, such as which midis were used
    and what stretch range. Instances are created by iterating through a folder containing midi files and concatenating
    their data into a single very large NoteArray instance, called master_note_array. Padding is added to the beginning
    of each midi file's pianoroll before concatenating to ensure the model is not predicting the next song from
    the previous song's notes, or there is at least enough space for the model to recognize the last song has ended.
    The master note array instance can be used for efficiently training conv1D neural nets on, as having one large note
    array to sample from ensures that longer songs are sampled more than shorter ones, producing a properly calibrated
    model.
    """

    def __init__(self,
                 path_to_directory_of_midi_files,
                 note_array_creator,
                 num_augmentations_per_midi_file=1,
                 stretch_range=[1.0, 1.0],
                 end_padding_timesteps=0,
                 timesteps_crop_range=None,
                 ):
        """
        path_to_directory_of_midi_files: String path to directory of midi files one level deep in file tree
        note_array_creator: NoteArrayCreator instance specifying how to crop and downsample a pianoroll into a NoteArray
        num_augmentations_per_midi_file: How many stretched pianorolls to create from each midi fil
        stretch_range: A tupe of two floats in range (0.0, infinity) specifying valid random range for stretch fractions
        end_padding_timesteps: How much padding in timesteps to add to the end of pianorolls before conccatenating
        timesteps_crop_range: Mostly for debugging - chop each pianoroll to be within timesteps_crop_range timesteps
        """

        self.path_to_directory_of_midi_files = path_to_directory_of_midi_files
        self.note_array_creator = note_array_creator
        self.num_augmentations_per_midi_file = num_augmentations_per_midi_file
        self.stretch_range = stretch_range
        self.end_padding_timesteps = end_padding_timesteps
        self.timesteps_crop_range = timesteps_crop_range

        self.midi_file_paths_list = get_midi_file_paths_list(self.path_to_directory_of_midi_files)

        self.note_array = self.get_master_note_array()

    def get_master_note_array(self):
        """
        Using the note arrays list generated in get_note_arrays_list, concatentate together into a single note array.
        """

        note_arrays_list = self.get_note_arrays_list()

        total_note_array_size = np.sum([note_array.get_length_in_notes() for note_array in note_arrays_list])

        master_note_array_values = np.zeros((total_note_array_size,)).astype('bool')

        concat_index = 0
        for note_array in note_arrays_list:
            master_note_array_values[concat_index:concat_index + note_array.get_length_in_notes()] = note_array.array

            concat_index += note_array.get_length_in_notes()

        return self.note_array_creator.get_instance(flat_array=master_note_array_values)

    def get_note_arrays_list(self):
        """
        Creates the master note array using the following steps:

        1. For each midi file in the list of midi files to use:
            a. Load file as Pianoroll instance pianoroll
            b. Trim silence off of ends of pianoroll
            c. Generate num_augmentations_per_midi_file NoteArrays using the following steps:
                i. Stretch pianoroll by random amount within prescribed range (using noisy even spacing method)
                ii. Add start and end padding to pianoroll
                iii. Create cropped and down-sampled NoteArray instance from this pianoroll using note_array_creator
            d. Add NoteArray instances to a list
        2. Concatenate the full list of NoteArray instances into a single master NoteArray instance by merging their
           array values.
        """

        note_arrays_list = []

        for midi_file_path in self.midi_file_paths_list:

            pianoroll = Pianoroll(midi_file_path)

            pianoroll.trim_silence_off_ends()

            if self.timesteps_crop_range != None:
                pianoroll = pianoroll[self.timesteps_crop_range[0]:self.timesteps_crop_range[1]]

            stretch_fractions = get_noisily_spaced_floats(start=self.stretch_range[0],
                                                          end=self.stretch_range[1],
                                                          num_points=self.num_augmentations_per_midi_file)

            for i in range(self.num_augmentations_per_midi_file):
                stretch_fraction = stretch_fractions[i]

                stretched_pianoroll = pianoroll.get_stretched(stretch_fraction=stretch_fraction)

                stretched_pianoroll.add_zero_padding(right_padding_timesteps=self.end_padding_timesteps)

                note_array = self.note_array_creator.get_instance(pianoroll=stretched_pianoroll)

                note_arrays_list.append(note_array)

        return note_arrays_list
