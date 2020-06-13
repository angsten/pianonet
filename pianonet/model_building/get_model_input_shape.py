def get_model_input_shape(model):
    """
    Returns the required (minimum) input size of a FCNN model by analyzing the conv1d layers.

    model: Fully convolutional neural net keras model for which the input shape is determined.
    """

    model_input_size = 0
    for layer in model.layers:
        if layer.name.find('conv') != -1:
            dilation_rate = layer.dilation_rate[0]
            kernel_size = layer.kernel_size[0]

            model_input_size += (kernel_size - 1) * max(dilation_rate, 2)

    return model_input_size
