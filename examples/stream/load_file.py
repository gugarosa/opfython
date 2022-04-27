from opfython.stream import loader

# Loading a .csv file
csv = loader.load_csv("data/boat.csv")

# Loading a .txt file
txt = loader.load_txt("data/boat.txt")

# Loading a .json file
json = loader.load_json("data/boat.json")
