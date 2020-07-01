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
# Model parameters should include either a 'filters' key with a list or list of lists describing the absolute numbers
# of filters in each block's layer, or a filter_increments list or list of lists describing the additional filters
# relative to the last layer.
# 'default_activation' should be used to specify the default activation function to use, such as 'relu' or 'elu', when
# a filter entry does not specifically specify one.
###

import pickle
import sys

import numpy as np

np.random.seed(0)  # should ensure consistent model weight initializations
from tensorflow.keras.layers import Input, Conv1D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant, he_normal


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

    if 'filters' in model_params:
        filters = model_params['filters']

        if not isinstance(filters[0], list):
            filters = [filters]
    else:
        filters = []
        filter_increments = model_params['filter_increments']

        if not isinstance(filter_increments[0], list):
            filter_increments = [filter_increments]

        filter_count = 0
        for filter_increments_list in filter_increments:
            filters_list = []
            for filter_increment in filter_increments_list:
                filter_count += filter_increment
                filters_list.append(filter_count)

            filters.append(filters_list)

    default_activation = model_params.get('default_activation', 'relu')

    def default_dilation_function(layer_index):
        return 2 ** layer_index

    use_calibrated_output_bias = model_params.get('use_calibrated_output_bias', False)

    if use_calibrated_output_bias:
        output_bias_initializer = Constant(-3.2) # This will default to P(note=1) = 0.04, which is a normal base rate
    else:
        output_bias_initializer = "zeros"

    default_kernel_initializer = model_params.get('default_kernel_initializer', 'glorot_uniform')

    if default_kernel_initializer == 'he_normal':
        default_initializer = he_normal

    ######################
    ### MODEL BUILDING ###
    ######################

    inputs = Input(shape=(None, 1))
    conv = inputs

    for block_id in range(0, len(filters)):
        block_filters = filters[block_id]

        for i in range(0, len(block_filters)):
            conv = Conv1D(filters=block_filters[i],
                          kernel_size=2,
                          strides=1,
                          dilation_rate=default_dilation_function(i),
                          padding='valid',
                          kernel_initializer=default_kernel_initializer)(conv)
            conv = Activation(default_activation)(conv)


    outputs = Conv1D(filters=1,
                     kernel_size=1,
                     strides=1,
                     dilation_rate=1,
                     padding='valid',
                     kernel_initializer=default_kernel_initializer,
                     bias_initializer=output_bias_initializer)(conv)
    outputs = Activation('sigmoid')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    model.save(model_output_path)


if __name__ == '__main__':
    main()
