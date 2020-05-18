###
#
# Usage: python master_note_array_creation.py /path/to/input/file.json /path/to/output/directory
#
# Description: Script for generating a master note array from input midi files or a directory of midi files. To specify
#              how to create the master note array instance, use a json file with the required parameters noted below.
#
#              The output master note array will be saved to
#
#                   /path/to/output/directory/prefix_name_in_json_{training, validation, full}_idx.mna
#
#              where idx is a counter updated to the next unique integer to avoid overwriting.
###

import json
import os
import random
import sys

from pianonet.core.midi_tools import get_midi_file_paths_list
from pianonet.core.note_array_transformer import NoteArrayTransformer
from pianonet.training_utils.master_note_array import MasterNoteArray


def main():
    arguments = sys.argv

    if len(arguments) != 3:
        print("Rerun with the proper arguments. Example usage:\n")
        print(" $ python master_note_array_creation.py /path/to/input/file.json /path/to/output/master_note_array.mna")
        print()
        return

    input_json_file_path = sys.argv[1]
    save_directory_path = sys.argv[2]

    with open(input_json_file_path, 'rb') as json_file:
        custom_parameters = json.load(json_file)

    print("\nGenerating MasterNoteArray using the following parameters:\n")
    for k, v in custom_parameters.items():
        print("\t" + str(k) + ": " + str(v))

    file_name_prefix = custom_parameters['file_name_prefix']

    min_key_index = custom_parameters['min_key_index']
    num_keys = custom_parameters['num_keys']
    resolution = custom_parameters['resolution']

    end_padding_range_in_seconds = custom_parameters['end_padding_range_in_seconds']
    num_augmentations_per_midi_file = custom_parameters['num_augmentations_per_midi_file']
    stretch_range = custom_parameters['stretch_range'] if (custom_parameters['stretch_range'] != []) else None
    time_steps_crop_range = custom_parameters['time_steps_crop_range'] if (
            custom_parameters['time_steps_crop_range'] != []) else None

    paths_to_directories_of_midi_files = custom_parameters['midi_locator']['paths_to_directories_of_midi_files']
    whitelisted_midi_file_names = custom_parameters['midi_locator']['whitelisted_midi_file_names']

    validation_fraction = custom_parameters['validation_fraction']
    training_fraction = 1.0 - validation_fraction

    midi_file_paths_list = []

    for path_to_directory_of_midi_files in paths_to_directories_of_midi_files:
        midi_file_paths_list += get_midi_file_paths_list(path_to_directory_of_midi_files)

    if len(whitelisted_midi_file_names) != 0:
        midi_file_paths_list = [file_path for file_path in midi_file_paths_list if
                                (os.path.basename(file_path) in whitelisted_midi_file_names)]

    total_midi_files_count = len(midi_file_paths_list)
    training_midi_files_count = int(training_fraction * total_midi_files_count)
    validation_midi_files_count = total_midi_files_count - training_midi_files_count

    using_validation_set = (validation_midi_files_count != 0)

    random.shuffle(midi_file_paths_list)

    midi_file_paths_split = {
        'training': midi_file_paths_list[0:training_midi_files_count],
        'validation': midi_file_paths_list[training_midi_files_count:]
    }

    for set_name in ['training', 'validation']:
        if (set_name == 'validation') and (not using_validation_set):
            print("\nNo files included in the validation master note array. Skipping its creation.")
            continue

        partial_midi_file_paths_list = midi_file_paths_split[set_name]

        print("\nGenerating {set_name} MasterNoteArray using the these midi file paths:".format(set_name=set_name))
        [print("\t" + path) for path in partial_midi_file_paths_list]

        note_array_transformer = NoteArrayTransformer(min_key_index=min_key_index,
                                                      num_keys=num_keys,
                                                      resolution=resolution)

        master_note_array = MasterNoteArray(
            midi_file_paths_list=partial_midi_file_paths_list,
            note_array_transformer=note_array_transformer,
            num_augmentations_per_midi_file=num_augmentations_per_midi_file,
            stretch_range=stretch_range,
            end_padding_range_in_seconds=end_padding_range_in_seconds,
            time_steps_crop_range=time_steps_crop_range,
        )

        i = 0
        while (True):

            if using_validation_set:
                set_name_string = "_" + set_name
            else:
                set_name_string = "_full"


            save_path = os.path.join(save_directory_path, file_name_prefix + "_" + str(i) + set_name_string + ".mna")

            if not os.path.exists(save_path):
                break

            i += 1

        print("Saving note array to file at " + save_path)

        master_note_array.save(file_path=save_path)


if __name__ == '__main__':
    main()
