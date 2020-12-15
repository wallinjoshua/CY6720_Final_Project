import sys
import csv
import numpy as np

# Get csv datafile, output filename, and column filename
if len(sys.argv) <= 3:
    print("Insufficient Command Line Arguments Given\n")
    print("Arguments should be: [input_file] [output_file] [columns_file]\n")
    exit()

input_datafile = str(sys.argv[1])
header = open(input_datafile, 'r').readline()
output_datafile = str(sys.argv[2])
columns_file = open(str(sys.argv[3]), 'r')

cols = []
for col in columns_file:
    if len(col.strip()) == 0 or col.strip()[0] == '#':
        continue
    cols.append(col.strip())

# Import datafile as 'data'
hdr = []
for l in header.split(','): 
    hdr.append(l.strip('"\n'))

track_cols = cols.copy()

i = 0
rel_cols = []
for lbl in cols:
    if lbl in hdr:
        rel_cols.append(hdr.index(lbl))
        track_cols.remove(lbl)
    i += 1
# for lbl in hdr:
#     if lbl in cols:
#         rel_cols.append(i)
#         track_cols.remove(lbl)
#     i += 1

if len(rel_cols) < len(cols):
    print("NOT ALL COLUMNS FOUND IN DATA.\n")
    for c in track_cols:
        print("Column '", c, "' not found in data\n")

data = np.genfromtxt(input_datafile, dtype='str', skip_header=1, delimiter=',')

num_items = len(data)
new_data = np.empty((num_items, 0))
for c in rel_cols:
    new_data = np.append(new_data, np.reshape(data[:,c], (num_items, -1)), axis=1)

writer = csv.writer(open(output_datafile + '.csv', 'w'))
writer.writerow(cols)
writer.writerows(new_data)