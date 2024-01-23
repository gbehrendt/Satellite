# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from matplotlib import pyplot as plt
import csv

file = "/home/gabe/code/cmake-build-debug/timing.csv"
file = "/home/gbehrendt/CLionProjects/Satellite/Timing/RK4/limited-memory/maxIter10000.csv"
 
# opening the CSV file
with open(file, mode ='r')as file:
   
  # reading the CSV file
  csvFile = csv.reader(file)
 
  timing = []
  contents = []
  read = False
  # displaying the contents of the CSV file
  for lines in csvFile:
    contents.append(lines)
    print(lines)

    if lines:
        if 'OverallAlgorithm....................: ' in lines[0]:
            line = lines[0]
            res = line.split('OverallAlgorithm....................:      ', 1)
            print(res[1])
            new = res[1]
            new1 = float(new.split(' ',1)[0])
            timing.append(new1)
            print(new1)
            print("FOUND Timing!!!!!!!!")
            


time = range(0,len(timing))
fig, ax = plt.subplots()
plt.bar(time,timing)

# ex = contents[41][0]
# print(ex)
# print('OverallAlgorithm' in ex)
