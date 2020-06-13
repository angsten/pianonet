###
#
# Usage: python script.py /path/to/parameters/file.json /path/to/model/output/directory
#
# Description: Script for generating a basic fully convolutional network with blocks but no pass through or gates. We
#              use padding='valid' instead of 'causal' to avoid non-contextual learning within training samples.
#
#              The output model will be saved to
#
#                   /path/to/model/output/directory
###

import pickle
import sys

import numpy as np

np.random.seed(0)  # should ensure consistent model weight initializations
from tensorflow.keras.layers import Input, Conv1D, Activation
from tensorflow.keras.models import Model


def main():
    arguments = sys.argv

    if len(arguments) != 3:
        print("Rerun with the proper arguments. Example usage:\n")
        print(" $ python script.py /path/to/parameters/file.json /path/to/model/output/directory")
        print()
        return

    print(sys.argv)

    parameters_file_path = sys.argv[1]
    model_output_path = sys.argv[2]

    with open(parameters_file_path, 'rb') as params_file:
        model_params = pickle.load(params_file)

    filters = np.array(model_params['filters'])

    if filters.ndim == 1:
        filters = np.array([filters])
    elif filters.ndim != 2:
        raise Exception("The 'filters' parameter must be a one or two dimensional array.")

    ######################
    ### MODEL BUILDING ###
    ######################

    inputs = Input(shape=(None, 1))
    conv = inputs

    for block_id in range(0, filters.shape[0]):
        block_filters = filters[block_id]

        for i in range(0, filters.shape[1]):
            conv = Conv1D(filters=block_filters[i], kernel_size=2, strides=1, dilation_rate=2 ** i, padding='valid')(
                conv)
            conv = Activation('relu')(conv)

    outputs = Conv1D(filters=1, kernel_size=1, strides=1, dilation_rate=1, padding='valid')(conv)
    outputs = Activation('sigmoid')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    model.save(model_output_path)


if __name__ == '__main__':
    main()
