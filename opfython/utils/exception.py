class ArgumentException(Exception):
    def __init__(self, argument):
        super().__init__("Missing input argument. Expects: " + "'" + argument + "'.")