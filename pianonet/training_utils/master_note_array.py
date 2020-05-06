import numpy as np

from pianonet.core.midi_tools import get_midi_file_paths_list
from pianonet.core.misc_tools import get_noisily_spaced_floats
from pianonet.core.note_array import NoteArray
from pianonet.core.pianoroll import Pianoroll


class MasterNoteArray(NoteArray):
    """
    A Notearray instance generated from a collection of NoteArrays loaded from a directory of midi files. The data in
    these files are concatenated into one note array with metadata tracked, such as which midis were used and what
    stretch range. These instances conveniently provide the training or validation data. MasterNoteArray instances can
    be saved and later reused to ensure consistent training input.

    Instances are created by iterating through a folder containing midi files and concatenating their data into a
    single very large NoteArray instance. Padding is added to the beginning of each midi file's pianoroll before
    concatenating to ensure the model is not predicting the next song from the previous song's notes, or there is
    at least enough space for the model to recognize the last song has ended. A master note array instance can be used
    for efficiently training conv1D neural nets on, as having one large note array to sample from ensures that longer
    songs are sampled more than shorter ones, producing a properly calibrated model.
    """

    def __init__(self,
                 file_path=None,
                 path_to_directory_of_midi_files=None,
                 note_array_transformer=None,
                 num_augmentations_per_midi_file=1,
                 stretch_range=[1.0, 1.0],
                 end_padding_timesteps=0,
                 timesteps_crop_range=None,
                 ):
        """
        file_path: Optional, can initialize by loading a previously saved master note array from disc.
        path_to_directory_of_midi_files: String path to directory of midi files one level deep in file tree
        note_array_transformer: NoteArrayTransformer instance specifying how to crop and downsample a pianoroll
        num_augmentations_per_midi_file: How many stretched pianorolls to create from each midi fil
        stretch_range: A tupe of two floats in range (0.0, infinity) specifying valid random range for stretch fractions
        end_padding_timesteps: How much padding in timesteps to add to the end of pianorolls before conccatenating
        timesteps_crop_range: Mostly for debugging - chop each pianoroll to be within timesteps_crop_range timesteps
        """

        if file_path != None:
            self.load(file_path=file_path)
        else:
            self.path_to_directory_of_midi_files = path_to_directory_of_midi_files
            self.note_array_transformer = note_array_transformer
            self.num_augmentations_per_midi_file = num_augmentations_per_midi_file
            self.stretch_range = stretch_range
            self.end_padding_timesteps = end_padding_timesteps
            self.timesteps_crop_range = timesteps_crop_range

            self.midi_file_paths_list = get_midi_file_paths_list(self.path_to_directory_of_midi_files)

            self.array = self.get_concatenated_flat_array()

    def get_concatenated_flat_array(self):
        """
        Using the flat arrays list generated in get_flat_arrays_list, concatentate together into a single flat array.
        """

        flat_arrays_list = self.get_flat_arrays_list()

        total_array_length = np.sum([flat_array.shape[0] for flat_array in flat_arrays_list])

        master_flat_array = np.zeros((total_array_length,), dtype='bool')

        concat_index = 0
        for i in range(len(flat_arrays_list)):
            flat_array = flat_arrays_list.pop(0) # we pop to reduce memory overhead
            master_flat_array[concat_index:concat_index + flat_array.shape[0]] = flat_array

            concat_index += flat_array.shape[0]

        return master_flat_array

    def get_flat_arrays_list(self):
        """
        Create the list of flat arrays from the midi files list using the following steps:

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

        flat_arrays_list = []

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

                flat_array = self.note_array_transformer.get_flat_array_from_pianoroll(pianoroll=stretched_pianoroll)

                flat_arrays_list.append(flat_array)

        return flat_arrays_list
