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

    if len(arguments) != 2:
        print("\nRerun with the proper arguments. Example usage:\n")
        print(" $ python runner.py /path/to/run/directory")
        print()
        return

    path_to_run_directory = sys.argv[1]

    run = Run(path_to_run_directory)
    run.execute()


if __name__ == '__main__':
    main()
