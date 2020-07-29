from tensorflow.keras.models import load_model

from pianonet.core.note_array import NoteArray
from pianonet.core.note_array_transformer import NoteArrayTransformer
from pianonet.model_inspection.performance_tools import get_performance


def get_performance_from_pianoroll(pianoroll_seed,
                              num_time_steps,
                              model_path):
    """
    Creates a performance starting from a pianoroll seed.
    """

    model = load_model(model_path)

    aversion_params_dict = {
        'probability_thresholds': [1.0, 1.0, 0.4, 0.05, 0.05, 0.05, 0.03, 0.03],
    }

    note_array_transformer = NoteArrayTransformer(
        min_key_index=31,
        num_keys=72,
    )

    pianoroll = pianoroll_seed
    pianoroll.add_zero_padding(left_padding_timesteps=48 * 10)
    seed_note_array = NoteArray(pianoroll=pianoroll, note_array_transformer=note_array_transformer)

    final_note_array = get_performance(model=model,
                    seed_note_array=seed_note_array,
                    num_time_steps=num_time_steps,
                    validation_fraction=0.0,
                    use_edge_aversion=True,
                    aversion_params_dict=aversion_params_dict,
                    assume_elu=True)

    final_pianoroll = final_note_array.get_pianoroll()

    final_pianoroll.trim_silence_off_ends()

    return final_pianoroll