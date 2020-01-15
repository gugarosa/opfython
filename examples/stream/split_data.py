import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_loader(txt)

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = s.split(
    X, Y, percentage=0.5, random_state=1)

# Merging data into a unique set
X, Y = s.merge(X_train, X_test, Y_train, Y_test)
