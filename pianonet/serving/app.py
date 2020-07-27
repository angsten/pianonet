from flask import Flask
from pianonet.model_inspection.performance_from_pianoroll import get_performance_from_pianoroll
from pianonet.core.pianoroll import Pianoroll

app = Flask(__name__)


@app.route('/')
def alive():
    print("YOOOOO")
    return 'OK'


@app.route('/performance', methods=['POST'])
def performance():
    """
    Expects the midi binary object in json body.
    seconds_of_performance is num seconds of new notes
    """

    midi_file_path = "app/pianonet/1_performance.midi"
    seconds_in_performance = 4

    final_pianoroll = get_performance_from_pianoroll(
        pianoroll_seed=Pianoroll(midi_file_path),
        num_time_steps=48*seconds_in_performance,
        model_path="app/pianonet/models/r9p0_3500kparams_approx_9_blocks_model"
    )

    final_pianoroll.save_to_midi_file('./tmp.midi')

    return {"http_code": 200, "code": "Success", "message": ""}




if __name__ == '__main__':
    app.run(host='0.0.0.0')