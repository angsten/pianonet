import sys; sys.path.append('../')

from pianonet.training_utils.master_note_array import MasterNoteArray
from pianonet.core.note_array_transformer import NoteArrayTransformer

save_path = '../data/master_note_arrays/little_bach_1.mna'

min_key_index = 34
num_keys = 64
resolution = 1.0

end_padding_time_steps = 96
stretch_range = (0.85, 1.2)
num_augmentations_per_midi_file = 1
time_steps_crop_range = (0, 4000)

note_array_transformer = NoteArrayTransformer(min_key_index=min_key_index, num_keys=num_keys, resolution=resolution)

master_note_array = MasterNoteArray(
    path_to_directory_of_midi_files="/Users/angsten/PycharmProjects/pianonet/pianonet/data/midi/bach/",
    note_array_transformer=note_array_transformer,
    num_augmentations_per_midi_file=num_augmentations_per_midi_file,
    stretch_range=stretch_range,
    end_padding_time_steps=end_padding_time_steps,
    time_steps_crop_range=time_steps_crop_range,
)


master_note_array.save(file_path=save_path)

