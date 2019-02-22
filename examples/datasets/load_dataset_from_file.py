from opfython.datasets.loaded import Loaded

# Saving the file's path
file_path = 'data/sample.json'

# Creating a loaded instance of dataset, where
# it will already process the external data to its structure
d = Loaded(file_path=file_path)

for i, sample in enumerate(d.samples):
    print(f'Sample: {i+1} | Label: {sample.label} | Features: {sample.features}.')