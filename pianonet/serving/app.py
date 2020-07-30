import os
import random

from flask import Flask, request, send_from_directory
from werkzeug.utils import secure_filename

from pianonet.core.pianoroll import Pianoroll
from pianonet.model_inspection.performance_from_pianoroll import get_performance_from_pianoroll

app = Flask(__name__)

base_path = "/app/"
# base_path = "/Users/angsten/PycharmProjects/pianonet"

performances_path = os.path.join(base_path, 'data', 'performances')

def get_random_midi_file_name():
    """
    Get a random midi file name that will not ever collide.
    """
    return str(random.randint(0, 10000000000000000000)) + ".midi"

def get_performance_path(midi_file_name):
    """
    Returns full path to performance midi file given a file name.
    """

    return os.path.join(performances_path, midi_file_name)

@app.route('/')
def alive():
    return 'OK'

@app.route('/performances/', methods=['GET'])
def get_performance():
    """
    Returns the requested performance as midi file.
    Expected query string is 'midi_file_name', such as 1234.midi
    """

    performance_midi_file_name = request.args.get('midi_file_name')
    performance_midi_file_name = secure_filename(performance_midi_file_name)
    print(performance_midi_file_name)

    if performance_midi_file_name == None:
        return {"http_code": 400, "code": "BadRequest", "message": "midi_file_name not found in request."}

    midi_file_path = get_performance_path(performance_midi_file_name)

    if not os.path.exists(midi_file_path):
        return {
            "http_code": 404,
            "code": "Not Found",
            "message": "midi_file " + performance_midi_file_name + " not found."
        }

    with open(midi_file_path, 'rb') as midi_file:
        return send_from_directory(performances_path, performance_midi_file_name)

@app.route('/create-performance', methods=['POST'])
def performance():
    """
    Expects post form data as follows:
        seed_midi_file_data: Midi file that forms the seed for a performance as string encoding like "8,2,3,4,5..."
        seconds_to_generate: Number of seconds of new notes to generate
        model_complexity: Quality of model to use, one of ['low', 'medium', 'high', 'highest']
    """

    seed_midi_file_data = request.form.get('seed_midi_file_data')

    if seed_midi_file_data == None:
        return {"http_code": 400, "code": "BadRequest", "message": "seed_midi_file_data not found in request."}
    else:
        seed_midi_file_int_array = [int(x) for x in seed_midi_file_data.split(',')]

        frame = bytearray()
        for i in seed_midi_file_int_array:
            frame.append(i)

        saved_seed_midi_file_path = os.path.join(base_path, 'data', 'seeds', get_random_midi_file_name())

        with open(saved_seed_midi_file_path, 'wb') as midi_file:
            midi_file.write(frame)

    seconds_to_generate = request.form.get('seconds_to_generate')

    if seconds_to_generate == None:
        return {"http_code": 400, "code": "BadRequest", "message": "seconds_to_generate not found in request."}
    else:
        seconds_to_generate = float(seconds_to_generate)

    model_complexity = request.form.get('model_complexity', 'low')

    if model_complexity == 'low':
        model_name = "micro_1"
    else:
        model_name = "r9p0_3500kparams_approx_9_blocks_model"

    model_path = os.path.join(base_path, 'models', model_name)

    input_pianoroll = Pianoroll(saved_seed_midi_file_path)
    input_pianoroll.trim_silence_off_ends()

    final_pianoroll = get_performance_from_pianoroll(
        pianoroll_seed=input_pianoroll,
        num_time_steps=int(48 * seconds_to_generate),
        model_path=model_path,
    )

    midi_file_name = get_random_midi_file_name()
    midi_file_path = get_performance_path(midi_file_name)
    final_pianoroll.save_to_midi_file(midi_file_path)

    return {"http_code": 200, "code": "Success", "message": "", "midi_file_name": midi_file_name}


if __name__ == '__main__':
    app.run(host='0.0.0.0')
