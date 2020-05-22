import logging
import pprint


class Logger(object):
    """
    Parent class that makes logging the output of derived classes much more convenient.
    """

    def __init__(self, logger_name, log_file_path):
        self.logger = logging.getLogger(logger_name)
        # f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s: %(message)s')
        f_handler = logging.FileHandler(log_file_path)
        f_handler.setFormatter(f_format)
        self.logger.addHandler(f_handler)
        self.logger.setLevel(logging.INFO)

        self.pp = pprint.PrettyPrinter(indent=4, width=100)

    def log(self, message=''):
        """
        Write message to the log file.

        message: Can be a string or dictionary that will get logged by self.logger.
        """

        if isinstance(message, str):
            self.logger.info(message)
        elif isinstance(message, dict):
            self.logger.info(self.pp.pformat(message))
