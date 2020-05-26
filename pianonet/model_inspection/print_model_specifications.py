from pianonet.model_building.get_model_input_shape import get_model_input_shape


def print_model_specifications(model, num_keys, print_function=None):
    """
    Prints the relevant specs of the model using the print_function

    model: Keras model to print the specifications of
    num_keys: Integer specifying how many keys are in the input piano time step states
    print_function: Method for logging or printing the specs. If None, print() is used.
    """

    if print_function == None:
        print_function = print

    print_function("")
    num_notes_in_model_input = get_model_input_shape(model)
    time_steps_receptive_field = num_notes_in_model_input / num_keys

    print_function("Number of notes in model input: " + str(num_notes_in_model_input))
    print_function("Time steps in receptive field: " + str(time_steps_receptive_field))
    print_function("Seconds in receptive field: " + str(round((time_steps_receptive_field) / 48, 2)))
    print_function("")
    model.summary(print_fn=print_function)
    print_function("")
