import pygame


def play_midi_from_file(midi_file_path='', multitrack=None, vol=0.5):
    """
    Playback midi from file on disk.
    """

    if midi_file_path == '':
        midi_file_path = './.tmp.mid'

        multitrack.write(midi_file_path)

    frequency = 44100
    bitsize = -16
    channels = 1
    buffer = 1024

    pygame.mixer.init(frequency, bitsize, channels, buffer)

    pygame.mixer.music.set_volume(vol)

    try:
        clock = pygame.time.Clock()

        try:
            pygame.mixer.music.load(midi_file_path)
            print("Music file {file_path} loaded for playback!".format(file_path=midi_file_path))
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
