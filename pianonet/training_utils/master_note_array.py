import numpy as np

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
    single very large NoteArray instance. Padding is added to the end of each midi file's pianoroll before
    concatenating to ensure the model is not predicting the next song from the previous song's notes, or there is
    at least enough space for the model to recognize the last song has ended. A master note array instance can be used
    for efficiently training conv1D neural nets on, as having one large note array to sample from ensures that longer
    songs are sampled more than shorter ones, producing a properly calibrated model.
    """

    def __init__(self,
                 file_path=None,
                 midi_file_paths_list=None,
                 note_array_transformer=None,
                 num_augmentations_per_midi_file=1,
                 stretch_range=None,
                 end_padding_time_steps=0,
                 time_steps_crop_range=None,
                 ):
        """
        file_path: Optional, can initialize by loading a previously saved master note array from disc
        midi_file_paths_list: List of string paths to midi files to load
        note_array_transformer: NoteArrayTransformer instance specifying how to crop and down-sample a pianoroll
        num_augmentations_per_midi_file: How many stretched pianoroll augmentations to create from each midi file
        stretch_range: A tuple of two floats in range (0.0, infinity) specifying the valid range for stretch fractions
        end_padding_time_steps: How much padding in time steps to add to the end of pianorolls before conccatenating
        time_steps_crop_range: Mostly for debugging - chop each pianoroll to be within time_steps_crop_range timesteps
        """

        if file_path != None:
            self.load(file_path=file_path)
        else:
            self.file_path = file_path
            self.midi_file_paths_list = midi_file_paths_list
            self.note_array_transformer = note_array_transformer
            self.num_augmentations_per_midi_file = num_augmentations_per_midi_file
            self.stretch_range = stretch_range if (stretch_range != None) else (1.0, 1.0)
            self.end_padding_time_steps = end_padding_time_steps
            self.time_steps_crop_range = time_steps_crop_range

            self.array = self.get_concatenated_flat_array()

    def get_concatenated_flat_array(self):
        """
        Take the flat arrays list generated in get_flat_arrays_list and concatenate together into a single flat array.
        """

        flat_arrays_list = self.get_flat_arrays_list()

        total_array_length = np.sum([flat_array.shape[0] for flat_array in flat_arrays_list])

        master_flat_array = np.zeros((total_array_length,), dtype='bool')

        concat_index = 0
        for i in range(len(flat_arrays_list)):
            flat_array = flat_arrays_list.pop(0)  # we pop to reduce memory overhead
            master_flat_array[concat_index:concat_index + flat_array.shape[0]] = flat_array

            concat_index += flat_array.shape[0]

        return master_flat_array

    def get_flat_arrays_list(self):
        """
        Create the list of flat arrays from the midi files list using the following steps:

        1. For each midi file in the list of midi files to use:
            a. Load file as Pianoroll instance pianoroll
            b. Trim silence off of ends of pianoroll
            c. Crop pianoroll to be within time_steps_crop_range time_steps
            c. Generate num_augmentations_per_midi_file NoteArrays using the following steps:
                i. Stretch pianoroll by random amount within prescribed range (using noisy even spacing method)
                ii. Add end padding to pianoroll
                iii. Create cropped and down-sampled NoteArray instance from this pianoroll using note_array_creator
            d. Add NoteArray instance's 1D array values of booleans to a list
        2. Concatenate the full list of 1D arrays into a single master flat array by concatenating their values.
        """

        flat_arrays_list = []

        for midi_file_path in self.midi_file_paths_list:

            print("\t==> Processing midi file at: " + midi_file_path)

            pianoroll = Pianoroll(midi_file_path)

            pianoroll.trim_silence_off_ends()

            if self.time_steps_crop_range != None:
                pianoroll = pianoroll[self.time_steps_crop_range[0]:self.time_steps_crop_range[1]]

            stretch_fractions = get_noisily_spaced_floats(start=self.stretch_range[0],
                                                          end=self.stretch_range[1],
                                                          num_points=self.num_augmentations_per_midi_file)

            for i in range(self.num_augmentations_per_midi_file):
                stretch_fraction = stretch_fractions[i]

                stretched_pianoroll = pianoroll.get_stretched(stretch_fraction=stretch_fraction)

                stretched_pianoroll.add_zero_padding(right_padding_timesteps=self.end_padding_time_steps)

                flat_array = self.note_array_transformer.get_flat_array_from_pianoroll(pianoroll=stretched_pianoroll)

                flat_arrays_list.append(flat_array)

        return flat_arrays_list
