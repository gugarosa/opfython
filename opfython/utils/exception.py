""" A generic exception module.
"""

class ArgumentException(Exception):
    """ An ArgumentException class to hold arguments exceptions.

        # Arguments
            argument: label of argument that needs an exception invoke.
    """
    def __init__(self, argument):
        super().__init__("Missing input argument. Expects: " + "'" + argument + "'.")