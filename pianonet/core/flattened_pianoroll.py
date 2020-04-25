import numpy as np
from pypianoroll import Multitrack, Track



class Pianoroll:
    """
    An wrapped array representing piano key states in time. The first axis represents the time step, each of which has
    an associated key state of 128 keys. This implementation represents keys as binary on or off states at each time
    step, so the array is a 2D set of booleans.

    Example pianoroll array state:

            t = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...

            C   0  1  1  1  1  0  0  0  0  0
            B   0  0  0  0  0  1  1  1  1  1
            A#  1  0  0  0  0  0  0  0  0  0
            A   0  1  1  1  1  0  0  0  0  0
            G#  0  0  0  0  0  0  0  0  0  0

    General MIDI notes: Note index 60 is middle C, program = 0 is piano instrument, the max velocity allowed is 128.
                        A midi file can store up to 16 different tracks.
    """

    def __init__(self, initializer):
        """
        initializer: A string that is a path to a midi file or an array of shape (time_steps, 128).
        """

        if isinstance(initializer, str):
            midi_file_path = initializer
            self.load_from_midi_file(midi_file_path)
        else:
            np_array = initializer
            self.array = np.copy(np_array)

    def load_from_midi_file(self, midi_file_path):
        """
        midi_file_path: String that is path to a midi file to load

        A merged and binarized numpy array (time_steps, 128) in shape is loaded into self.array.
        """

        multitrack = Multitrack(filename=midi_file_path)

        multitrack.check_validity()

        multitrack.merge_tracks(track_indices=[i for i in range(len(multitrack.tracks))], mode='max', program=0,
                                remove_merged=True)

        pianoroll_array = (multitrack.tracks[0].pianoroll > 0.0).astype('bool')

        self.array = pianoroll_array








