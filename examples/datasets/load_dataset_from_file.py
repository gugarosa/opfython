from opfython.datasets.loaded import Loaded

# Saving the file's path
file_path = 'a.csv'

# Creating a loaded instance of dataset, where
# it will already process the external data to its structure
d = Loaded(file_path=file_path)

print(d.samples[2].features)