from pianonet.training_utils.master_note_array import MasterNoteArray
from pianonet.core.note_array_transformer import NoteArrayTransformer

from pianonet.core.midi_tools import get_midi_file_paths_list

save_path = '../data/master_note_arrays/little_bach_1.mna'
path_to_directory_of_midi_files = '../data/midi/bach/'

import os
print(os.path.abspath(path_to_directory_of_midi_files))

min_key_index = 34
num_keys = 64
resolution = 1.0

end_padding_time_steps = 96
stretch_range = (0.85, 1.2)
num_augmentations_per_midi_file = 1
time_steps_crop_range = (0, 4000)

note_array_transformer = NoteArrayTransformer(min_key_index=min_key_index, num_keys=num_keys, resolution=resolution)

midi_file_paths_list = get_midi_file_paths_list(path_to_directory_of_midi_files)

print("Generatign note array using the following file paths:")
[print(path) for path in midi_file_paths_list]

master_note_array = MasterNoteArray(
    midi_file_paths_list=midi_file_paths_list,
    note_array_transformer=note_array_transformer,
    num_augmentations_per_midi_file=num_augmentations_per_midi_file,
    stretch_range=stretch_range,
    end_padding_time_steps=end_padding_time_steps,
    time_steps_crop_range=time_steps_crop_range,
)

master_note_array.save(file_path=save_path)
