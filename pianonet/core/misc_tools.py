import hashlib
import json

import numpy as np


def get_noisily_spaced_floats(start, end, num_points):
    """
    start: starting float of range
    end: ending float of range
    num_points: number of floats to output

    Returns evenly spaced floats from start to end inclusive with random noise added to each point.
    Example: get_noisily_spaced_floats(start=0.8, end=1.2, num_points=3) could give
             the result array([0.84758304, 1.04413777, 1.18226883])
    """

    evenly_spaced_points = np.linspace(start=start, stop=end, num=(num_points + 1), endpoint=True)[:-1]

    noise_to_add_array = np.random.random(num_points) * ((end - start) / num_points)

    return evenly_spaced_points + noise_to_add_array


def get_hash_string_of_numpy_array(array):
    """
    Returns a 32 character long string that is a deterministic hash of a numpy array's data.
    """

    return hashlib.md5(array.data.tobytes()).hexdigest()


def save_dictionary_to_json_file(dictionary, json_file_path):
    """
    Saves dictionary to file at json_file_path.
    """

    with open(json_file_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)


def load_dictionary_from_json_file(json_file_path):
    """
    Loads a dictionary from file at json_file_path, returns the dictionary.
    """

    with open(json_file_path, 'r') as json_file:
        return json.load(json_file)
