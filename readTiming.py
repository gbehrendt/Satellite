import csv
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import linalg as LA
from itertools import zip_longest
import re
import itertools as it

# Define a function 'get_avg' that calculates the average of a list, ignoring 'None' values.
def get_avg(x):
    # Remove 'None' values from the list 'x'.
    x = [i for i in x if i is not None]
    # Calculate the average of the remaining values.
    return sum(x, 0.0) / len(x)

hessianApprox = "limited-memory"
ts = "10"
iters = ["4","5","10","20","50","100"]
itersDict = {}

for maxIter in iters:
    readFolder = "/home/gbehrendt/CLionProjects/untitled/Timing2/Euler/" + hessianApprox + "/ts" + ts + "/maxIter" + maxIter + "/"
    
    masterDict = {}
    
    directory = os.fsencode(readFolder)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            readFile = directory.decode("utf-8") + filename
            # print(readFile)
            
            with open(readFile, mode='r') as file:
                # Create a CSV reader with DictReader
                csv_reader = csv.reader(file)
                trialNum = re.findall(r"\d+", filename)
                
                timing = []
                contents = []
                read = False
                # displaying the contents of the CSV file
                for lines in csv_reader:
                  contents.append(lines)
                  #print(lines)
    
                  if lines:
                      if 'OverallAlgorithm....................: ' in lines[0]:
                          line = lines[0]
                          res = line.split('OverallAlgorithm....................:      ', 1)
                          new = res[1]
                          new1 = float(new.split(' ',1)[0])
                          timing.append(new1)
    
                numLoops = len(timing)
                tt = np.arange(numLoops)
                
                trialDict = {'trial':filename, 'trialNum':trialNum[0], 'timing':timing, 'numLoops':numLoops, 'tt':tt}
                masterDict[trialNum[0]] = trialDict
    
    # Make list of timing lists
    allTimings = []
    for key in masterDict:
        allTimings.append(masterDict[key]['timing'])
    
    avgTiming= list(map(get_avg, it.zip_longest(*allTimings)))
    avgTimescale = np.arange(len(avgTiming))
    
    itersDict[maxIter] = {'avgTiming':avgTiming, 'avgTimescale':avgTimescale}

    
    # fig1, ax1 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax1.plot(masterDict[key]['tt'], masterDict[key]['timing'], label = key)
    # ax1.set_xlabel("Time $(s)$", fontsize =14)
    # ax1.set_title("Timing", fontsize =14)
    # ax1.grid()
    # plt.show()
    
fig2, ax2 = plt.subplots()
plt.style.use('default')
for key in itersDict:
    ax2.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgTiming'], label = key)
ax2.set_ylabel("IpOpt Algorithm Time $(s)$", fontsize =14)
ax2.set_xlabel("Loop #", fontsize =14)
ax2.set_title(r"Average Timing", fontsize =14)
ax2.legend(fontsize =12, title="Maximum Iterations")
ax2.grid()
plt.show()


# file = "/home/gbehrendt/CLionProjects/Satellite/testTiming/RK4/limited-memory/ts5/maxIter4/trial103.csv"
 
# # opening the CSV file
# with open(file, mode ='r')as file:
   
#   # reading the CSV file
#   csvFile = csv.reader(file)
 
#   timing = []
#   contents = []
#   read = False
#   # displaying the contents of the CSV file
#   for lines in csvFile:
#     contents.append(lines)
#     #print(lines)

#     if lines:
#         if 'OverallAlgorithm....................: ' in lines[0]:
#             line = lines[0]
#             res = line.split('OverallAlgorithm....................:      ', 1)
#             #print(res[1])
#             new = res[1]
#             new1 = float(new.split(' ',1)[0])
#             timing.append(new1)
#             #print(new1)
#             #print("FOUND Timing!!!!!!!!")
            


# time = range(0,len(timing))
# fig, ax = plt.subplots()
# plt.bar(time,timing)

# ex = contents[41][0]
# print(ex)
# print('OverallAlgorithm' in ex)
