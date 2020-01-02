import opfython.utils.constants as c
import opfython.utils.logging as l
from opfython.core.heap import Heap
from opfython.core.opf import OPF
from opfython.core.subgraph import Subgraph

logger = l.get_logger(__name__)


class SupervisedOPF(OPF):
    """A SupervisedOPF which implements the supervised version of OPF classifier.

    References:
        J. P. Papa, A. X. Falcão and C. T. N. Suzuki. Supervised Pattern Classification based on Optimum-Path Forest. International Journal of Imaging Systems and Technology (2009).

    """

    def __init__(self, pre_computed_distance=False):
        """Initialization method.

        Args:
            pre_computed_distance (bool): Whether OPF should use pre-computed distances or not.

        """

        logger.info('Overriding class: OPF -> SupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(SupervisedOPF, self).__init__(
            pre_computed_distance=pre_computed_distance)

        logger.info('Class overrided.')

    def fit(self, X, Y):
        """Fits data in the classifier.

        Args:
            X (np.array): Array of features.
            Y (np.array): Array of labels.

        """

        #
        h = Heap()

        # Creating a subgraph
        self.g = Subgraph(X, Y)

        # Finding prototypes
        self._find_prototypes(self.g)

        #
        costs = np.zeros(self.g.n_nodes)

        #
        costs.fill(c.FLOAT_MAX)

        #
        for i in range(self.g.n_nodes):
            if self.g.nodes[i].status == c.PROTOTYPE:
                self.g.nodes[i].pred = c.NIL
                costs[i] = 0
                self.g.nodes[i].predicted_label = self.g.nodes[i].label
                h.insert(i)

        i = 0

        while is not h.empty():
            p = h.remove()
            self.g.idx_nodes.append(p)
            i += 1
            self.g.nodes[p].cost = costs[p]

            for q in range(self.g.n_nodes):
                if not p == q:
                    if costs[p] < costs[q]:
                        if self.pre_computed_distance:
                            weight = self.distances[g.nodes[p].idx][g.nodes[q].idx]
                        else:
                            weight = d.log_euclidean_distance(
                                g.nodes[p].features, g.nodes[q].features)
                            

    def predict(self, X):
        """Predicts new data using the pre-trained classifier.

        Args:
            X (np.array): Array of features.

        Returns:
            A list of predictions for each record of the data.

        """

        pass
