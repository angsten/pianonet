import copy
import math
import random
import time
from collections import deque

import numpy as np
from scipy.linalg.blas import sgemm as matmul
from tensorflow.keras import backend as K

from pianonet.model_building.get_model_input_shape import get_model_input_shape


def get_performance(model,
                    seed_note_array,
                    num_time_steps,
                    validation_fraction=0.0,
                    use_edge_aversion=False,
                    aversion_params_dict=None,
                    assume_elu=False):
    """
    Takes in a seed note array and generated num_timesteps of piano notes sampled
    from the model's output probabilities. A full NoteArray instance, including
    the seed note array data, is returned.

    seed_note_array: Seed data for model input. If None, silence is used as the seed. NOTE! This note array is
                     assumed to have its keys properly aligned. That is, indices 0, num_keys, 2*num_keys, ...etc.
                     are at the starts of new time steps, and a full key state is between 0 and num keys, for
                     instance.
    model: The Keras trained model for generating the probabilities of new notes
    num_time_steps: How many new time steps of notes to generate using the model
    validation_fraction: Float between 0 and 1 specifying the fraction of predicted notes for which the optimized
                         model output will be randomly compared to the model.predict output every
                         validation_step_size notes. Set to 0.0 for no validation (only necessary for debugging).
    use_edge_aversion: Boolean controlling whether or not to keep notes biased toward the middle of the output space.
                       This will tend to prevent the outputs from 'going over the edge', which can cause odd sounding
                       performances.
    aversion_params_dict: Params to control how strong edge aversion is
    assume_elu: If true, optimize to use numpy based elu with alpha = 1 for internal hidden activations (2X faster)
    """

    num_keys = seed_note_array.note_array_transformer.num_keys

    num_notes_in_model_input = get_model_input_shape(model)

    input_placeholder = model.input
    output_placeholders = [layer.output for layer in model.layers]
    functor = K.function([input_placeholder], output_placeholders)

    input_data = seed_note_array.get_values_in_range(start_index=-num_notes_in_model_input,
                                                     end_index=None,
                                                     use_zero_padding_for_out_of_bounds=False)

    layer_outputs = functor([np.array([input_data.reshape(-1, 1)])])

    def get_initial_state_at(layer_index, num_states):
        """
        Get the num_states outputs from layer layer_index.
        """

        intermediate_output = layer_outputs[layer_index]
        return [np.transpose([state]) for state in intermediate_output[0][-num_states - 1:-1]]

    print("Initializing state queues.")

    num_model_layers = len(model.layers)

    initial_state_queues = []

    for i in range(0, num_model_layers - 2):
        layer = model.layers[i]

        if (layer.name.find('conv1d') != -1) and (i > 1):
            initial_state_queues.append(
                deque(get_initial_state_at(layer_index=(i - 1), num_states=layer.dilation_rate[0])))
        else:
            initial_state_queues.append(None)

    print("Resetting the state queues to the initial state (for a new performance).\n")
    state_queues = copy.deepcopy(initial_state_queues)  # Only run when starting a new performance
    output_data = seed_note_array.array.copy().tolist()

    raw_input = deque(copy.deepcopy(output_data)[-num_notes_in_model_input:])
    input_end_index = len(raw_input) - 1

    # This assumes a kernel size of two
    w_at_one = np.transpose(model.get_layer(index=1).get_weights()[0])
    b_at_one = np.transpose([model.get_layer(index=1).get_weights()[1]])

    saved_activation_functions = [None]
    for i in range(1, num_model_layers):
        saved_activation_functions.append(model.layers[i].activation)

    saved_weight_entries = []
    dilation_rates = []
    for i in range(0, num_model_layers - 2):
        node = model.layers[i]
        dilation_rates.append(0)
        if node.name.find('conv1d') != -1:
            dilation_rates[-1] = node.dilation_rate[0]
            weights = node.get_weights()
            w = weights[0]
            b = np.transpose([weights[1]])

            w1 = np.transpose(w[0])
            w2 = np.transpose(w[1])

            saved_weight_entries.append({
                'w1': w1,
                'w2': w2,
                'b': b,
            }
            )
        else:
            saved_weight_entries.append({})

    dilation_rates.append(model.layers[-2].dilation_rate[0])

    final_layer = model.layers[num_model_layers - 2]
    final_weights = final_layer.get_weights()
    w_final = np.transpose(final_weights[0][0])
    b_final = np.transpose(np.array([[final_weights[1]]]))

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    def get_activated(x, layer_index):
        if assume_elu:
            return np.where(x > 0, x, (np.exp(x) - 1))
        else:
            activation_function = saved_activation_functions[layer_index + 1]
            return activation_function(x)


    def get_output(input_position):

        inputs = np.transpose([raw_input[input_position - 1], raw_input[input_position]])

        right_input = get_activated(matmul(1.0, w_at_one, inputs) + b_at_one, layer_index=1)

        for layer_index in range(3, num_model_layers - 3, 2):

            layer_state_queue = state_queues[layer_index]

            w1 = saved_weight_entries[layer_index]['w1']
            w2 = saved_weight_entries[layer_index]['w2']
            b = saved_weight_entries[layer_index]['b']

            left_input = layer_state_queue.popleft()
            layer_state_queue.append(right_input)

            right_input = matmul(1.0, w1, left_input) + matmul(1.0, w2, right_input) + b

            right_input = get_activated(right_input, layer_index)

        final_result = matmul(1.0, w_final, right_input) + b_final

        final_result = sigmoid(final_result)  # this assumes sigmoid!

        return final_result



    start = time.time()

    seconds = -1
    for time_step in range(0, num_time_steps):

        if time_step % 48 == 0:
            seconds += 1
            print("==> Time step " + str(time_step) + " seconds of audio is " + str(seconds))

        for key in range(0, num_keys):

            probability_of_key_played = get_output(input_end_index)

            if random.uniform(0.0, 1.0) < validation_fraction:
                res_model = model.predict([[np.array(raw_input).reshape(1, -1)]])[0][-1]

                optimized_inconsistency_magnitude = abs(res_model - probability_of_key_played)
                if optimized_inconsistency_magnitude > 1e-6:
                    print("  Warning: Optimized output mode giving inconsistent results.")
                    print("    Difference is " + str(optimized_inconsistency_magnitude))

            if use_edge_aversion:
                distance_in_keys_from_edge = min(key, (num_keys - 1) - key)

                if distance_in_keys_from_edge < len(aversion_params_dict['probability_thresholds']):
                    probability_threshold = aversion_params_dict['probability_thresholds'][distance_in_keys_from_edge]

                    if probability_of_key_played < probability_threshold:
                        probability_of_key_played = 0.0

            prediction = (probability_of_key_played > random.uniform(0.0, 1.0))
            output_data.append(prediction)

            raw_input.popleft()
            raw_input.append(prediction)

        end = time.time()

    print("\nTime per second of audio:", round((end - start) / (num_time_steps / 48), 3), "seconds")

    outputs_added = len(output_data) - seed_note_array.get_length_in_notes()

    print("Timesteps added:", outputs_added / num_keys)

    final_output_note_array = seed_note_array.note_array_transformer.get_note_array(flat_array=np.array(output_data))

    return final_output_note_array
