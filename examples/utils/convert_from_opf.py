import opfython.utils.converter as c

# Defining the input OPF file
opf_file = 'data/boat.dat'

# Converting from .dat or .opf to .txt
c.opf2txt(opf_file, output_file='out.txt')

# Converting from .dat or .opf to .csv
c.opf2csv(opf_file, output_file='out.csv')

# Converting from .dat or .opf to .json
c.opf2json(opf_file, output_file='out.json')
