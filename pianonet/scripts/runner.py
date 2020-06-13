###
#
# Usage: python runner.py /path/to/run/directory
#
# Description: Script for running a directory representing a Run instance.
#
###

import sys

from pianonet.training_utils.run import Run


def main():
    arguments = sys.argv

    if len(arguments) not in (2, 3):
        print("\nRerun with the proper arguments. Example usage:\n")
        print(" $ python runner.py /path/to/run/directory train")
        print()
        return

    path_to_run_directory = arguments[1]

    if len(arguments) == 3:
        mode = arguments[2]
    else:
        mode = 'train'


    run = Run(path_to_run_directory, mode=mode)


if __name__ == '__main__':
    main()
