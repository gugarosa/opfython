import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as l

logger = l.get_logger(__name__)


class Set:
    """A standard implementation of a Set structure.

    """

    def __init__(self):
        """Initialization method.

        """

        # Size of the set
        self.size = 0

    @property
    def size(self):
        """int: Size of the set.

        """

        return self._size

    @size.setter
    def size(self, size):
        if not isinstance(size, int):
            raise e.TypeError('`size` should be an integer')
        if size < 0:
            raise e.ValueError('`size` should be >= 0')

        self._size = size

    def insert(self, p):
        """Inserts a new element into the set.

        Args:
            p (int): Element to be inserted.

        """

        # Incrementing the size of the set
        self.size += 1

    def remove(self):
        """Removes an element from the set.

        Returns:
            The value of the removed element.

        """

        # Decrementing the size of the set
        self.size -= 1
