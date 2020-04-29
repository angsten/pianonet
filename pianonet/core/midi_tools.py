import tempfile
import os

import pygame


def play_midi_from_file(midi_file_path='', multitrack=None, vol=1.0):
    """
    midi_file_path: path to midi file on disc
    multitrack: pypianoroll style multitrack instance that can be written to file
    vol: How loudly to play, from 0.0 to 1.0

    Play back midi data over computer audio by reading from file on disk or a pypianoroll multitrack instance.
    """

    midi_file = midi_file_path

    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=1024)

    pygame.mixer.music.set_volume(vol)

    with tempfile.TemporaryFile() as temporary_midi_file:

        if midi_file_path == '':
            multitrack.to_pretty_midi().write(temporary_midi_file)
            temporary_midi_file.seek(0)

            midi_file = temporary_midi_file

        try:
            clock = pygame.time.Clock()

            try:
                pygame.mixer.music.load(midi_file)
            except pygame.error:
                print("File {file_path} not found:".format(file_path=midi_file_path), pygame.get_error())
                return

            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                clock.tick(30)

        except KeyboardInterrupt:
            pygame.mixer.music.fadeout(1000)
            pygame.mixer.music.stop()
            raise SystemExit


def is_midi_file(file_name):
    """
    Returns true if file_name string looks like a midi file.
    """

    return file_name.lower().find(".mid") != -1


def get_midi_file_paths_list(directory_path):
    """
    directory_path: Path to directory containing midi files

    Only one level in the file tree is considered. A list of absolute paths to the midi files is returned.
    """

    if not os.path.isdir(directory_path):
        raise Exception(str(directory_path) + " is not a directory.")

    midi_file_names_in_directory = [file_name for file_name in os.listdir(directory_path) if is_midi_file(file_name)]

    midi_file_paths_list = [os.path.abspath(os.path.join(directory_path, file_name)) for file_name in
                            midi_file_names_in_directory]

    return midi_file_paths_list
