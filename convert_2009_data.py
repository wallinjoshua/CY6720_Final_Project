import sys
import numpy as np
import csv

data = np.genfromtxt('student_data_2009.csv', delimiter=',', skip_header=1, dtype=float)
new_data = np.zeros((len(data),12), dtype=float)

new_header = ['STU_ID', 'BYSEX', 'BYENGLSE', 'BYSES1', 'BYSES2', 'BYS24F', 'BYGRDRPT', 'BYLGRRPT', 'BYTXMSTD', 'BYPISAME', 'BYACTCTL', 'F2HSSTAT']

new_data[:,0] = data[:,0]
new_data[:,1] = data[:,1]

# CONVERT SCIENCE AND MATH SELF-EFFICACY SCORE
data[:,2] = (((data[:,2] + 9.0)/(1.83+9.0))*(1.62+9)) - 9
mean_scores = np.sum(data[:,[2,4]], axis=1)/2.0
engl_self_efficacy = (((mean_scores + 2.197)/(1.596+2.197))*(1.596+2.197)) - 2.197
new_data[:,2] = engl_self_efficacy

# CONVERT SOCIO-ECONOMIC STATUS
new_data[:,3] = (((data[:,4] + 8.0)/(2.8807+8.0))*(1.82+2.11)) - 2.11
new_data[:,4] = (((data[:,4] + 8.0)/(2.8807+8.0))*(1.98+2.11)) - 2.11

# REPLACE SUSPENSIONS COLUMN WITH ALL -4
new_data[:,5] = -4.0

# GET NUMBER OF REPEATED GRADES AND LATEST
for r in range(len(data)):
    last_i = -3
    for i in range(5,16):
        data[r][i] = 0 if data[r][i] < 0 else data[r][i]
        if data[r][i] >= 1:
            last_i = i
    new_data[i][7] = i - 5
for i in range(len(data)):
    new_data[i][6] = 0 if sum(data[i,[5,16]]) < 1 else 1 if sum(data[i, [5,16]]) < 2 else 2

# CONVERT MATH STANDARDIZED TEST SCORE
new_data[:,8] = (((data[:,17] + 8.0)/(3.0283+8.0))*(86.68-19.38)) + 19.38

# REPLACE PISA COLUMN WITH ALL 497.2329404731924
new_data[:,9] = 497.232940

# CONVERT MATH AND SCIENCE EFFORT SCORE
new_data[:,10] = np.sum(data[:,[18,19]], axis=1)/2.0
new_data[:,11] = (((new_data[:,9] + 9.0)/(1.18+9.0))*(1.702+2.424)) - 2.424

# BINARIZE HIGH SCHOOL COMPELTION
new_col = []
for v in data[:,10]:
    if v in [1]:
        new_col.append(1)
    else:
        new_col.append(0)

new_data[:,11] = np.array(new_col)

writer = csv.writer(open('converted_student_data_2009' + '.csv', 'w'))
writer.writerow(new_header)
writer.writerows(new_data)