import logging
import argparse

class ArgRange(object):
    """
    Define a class to impose a range for float/int arguments in the argument parser
    """
    def __init__(self, start=None, end=None):
        """
        Args:
            start (float, optional): Minimum value for the range. If None, there is no lower bound. Defaults to None.
            end (float, optional): Maximum value for the range. If None, there is no upper bound. Defaults to None.
        """
        self.start = start
        self.end = end
        assert not (start is None and end is None)

    def __eq__(self, other):
        if self.start is None:
            return other <= self.end
        if self.end is None:
            return self.start <= other
        return self.start <= other <= self.end


_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']

def _log_level_string_to_int(log_level_string):
    if not log_level_string in _LOG_LEVEL_STRINGS:
        message = 'invalid choice: {0} (choose from {1})'.format(log_level_string, _LOG_LEVEL_STRINGS)
        raise argparse.ArgumentTypeError(message)

    log_level_int = getattr(logging, log_level_string, logging.INFO)
    # check the logging log_level_choices have not changed from our expected values
    assert isinstance(log_level_int, int)

    return log_level_int