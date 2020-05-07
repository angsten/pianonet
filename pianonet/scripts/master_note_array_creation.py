###
#
# Usage: python master_note_array_creation.py /path/to/input/file.json /path/to/output/directory
#
# Description: Script for generating a master note array from input midi files or a directory of these files. To specify
#              how to create the master note array instance, use a json file with the required parameters noted below.
#              The output master note array will be saved to /path/to/output/directory/prefix_name_in_json_idx.mna,
#              where idx is a counter updated to the next unique integer to avoid overwriting.
###

import sys
import json

from pianonet.core.midi_tools import get_midi_file_paths_list
from pianonet.core.note_array_transformer import NoteArrayTransformer
from pianonet.training_utils.master_note_array import MasterNoteArray


def main():
    arguments = sys.argv

    if len(arguments) != 3:
        print("Rerun with the proper arguments. Usage:")
        print("python master_note_array_creation.py /path/to/input/file.json /path/to/output/master_note_array.mna")

        return

    input_json_file_path = sys.argv[1]
    save_path = sys.argv[2]

    custom_parameters = json.load(input_json_file_path)

    print(custom_parameters)

    return

    path_to_directory_of_midi_files = '../data/midi/bach/'

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


if __name__ == '__main__':
    main()
