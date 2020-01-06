import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
from opfython.core.subgraph import Subgraph
from opfython.models.supervised import SupervisedOPF

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_array(txt)

# Creates a SupervisedOPF instance
opf = SupervisedOPF()

# Fits training data into the classifier
opf.fit(X, Y)

# Predicts new data
preds = opf.predict(X)

print(preds)

g.opf_accuracy(Y, preds)
