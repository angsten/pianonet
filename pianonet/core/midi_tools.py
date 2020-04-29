import tempfile

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
