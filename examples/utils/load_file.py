import opfython.utils.loader as l

# Loading a .csv file
csv = l.load_csv('data/sample.csv')

# Loading a .txt file
txt = l.load_txt('data/sample.txt')

# Loading a .json file
json = l.load_json('data/sample.json')

print(csv)
print(txt)
