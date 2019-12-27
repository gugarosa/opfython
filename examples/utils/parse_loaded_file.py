import opfython.utils.loader as l
import opfython.utils.parser as p

# Loading a .txt file to a data frame
txt = l.load_txt('data/sample.txt')

# Parsing a pre-loaded data frame
ids, labels, features = p.parse_df(txt)

print(ids, labels, features)