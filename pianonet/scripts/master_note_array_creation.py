###
#
# Usage: python master_note_array_creation.py /path/to/input/file.json /path/to/output/directory
#
# Description: Script for generating a master note array from input midi files or a directory of these files. To specify
#              how to create the master note array instance, use a json file with the required parameters noted below.
#              The output master note array will be saved to /path/to/output/directory/prefix_name_in_json_idx.mna,
#              where idx is a counter updated to the next unique integer to avoid overwriting.
###

import json
import os
import sys
import random

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

    print("\nGenerating master note array using the following parameters:")
    for k, v in custom_parameters.items():
        print("\t" + str(k) + ": " + str(v))

    file_name_prefix = custom_parameters['file_name_prefix']

    min_key_index = custom_parameters['min_key_index']
    num_keys = custom_parameters['num_keys']
    resolution = custom_parameters['resolution']

    end_padding_time_steps = custom_parameters['end_padding_time_steps']
    stretch_range = custom_parameters['stretch_range']
    num_augmentations_per_midi_file = custom_parameters['num_augmentations_per_midi_file']

    if custom_parameters['time_steps_crop_range'] != []:
        time_steps_crop_range = custom_parameters['time_steps_crop_range']
    else:
        time_steps_crop_range = None

    if custom_parameters['stretch_range'] != []:
        stretch_range = custom_parameters['stretch_range']
    else:
        stretch_range = None

    path_to_directory_of_midi_files = custom_parameters['midi_locator']['path_to_midi_file_directory']
    whitelisted_midi_file_names = custom_parameters['midi_locator']['whitelisted_midi_file_names']

    validation_fraction = custom_parameters['validation_fraction']
    training_fraction = 1.0 - validation_fraction

    midi_file_paths_list = get_midi_file_paths_list(path_to_directory_of_midi_files)

    if len(whitelisted_midi_file_names) != 0:
        midi_file_paths_list = [file_path for file_path in midi_file_paths_list if
                                (os.path.basename(file_path) in whitelisted_midi_file_names)]

    total_midi_files_count = len(midi_file_paths_list)
    training_midi_files_count = int(training_fraction * total_midi_files_count)
    validation_midi_files_count = total_midi_files_count - training_midi_files_count

    random.shuffle(midi_file_paths_list)

    midi_file_paths_split = {
        'training': midi_file_paths_list[0:training_midi_files_count],
        'validation': midi_file_paths_list[
                      training_midi_files_count:training_midi_files_count + validation_midi_files_count]
    }

    for set_name in ['training', 'validation']:
        if (set_name == 'validation') and (validation_midi_files_count == 0):
            print("No files included in the validation master note array. Skipping its creation.")
            continue

        set_midi_file_paths_list = midi_file_paths_split[set_name]

        print("\nGenerating {set_name} master note array using the these midi file paths:".format(set_name=set_name))
        [print("\t" + path) for path in set_midi_file_paths_list]

        note_array_transformer = NoteArrayTransformer(min_key_index=min_key_index, num_keys=num_keys,
                                                      resolution=resolution)

        master_note_array = MasterNoteArray(
            midi_file_paths_list=set_midi_file_paths_list,
            note_array_transformer=note_array_transformer,
            num_augmentations_per_midi_file=num_augmentations_per_midi_file,
            stretch_range=stretch_range,
            end_padding_time_steps=end_padding_time_steps,
            time_steps_crop_range=time_steps_crop_range,
        )

        i = 0
        while (True):
            save_path = os.path.join(save_directory_path, file_name_prefix + "_" + set_name + "_" + str(i) + ".mna")

            if not os.path.exists(save_path):
                break

            i += 1

        print("Saving note array to file at " + save_path)

        master_note_array.save(file_path=save_path)


if __name__ == '__main__':
    main()
