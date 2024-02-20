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

whichFolder = 1
simulationType = 0

samplingTime = "10"
iters = ["5","10","20","50","100"]
iters = ["5","6","7","8","9","10","100","1000"]
# iters = ["5","6","7","8","9","10","15","50","100","1000"]
iters = ["5","6","7","8","9","10","15","50","100","10000"]
# iters = ["10000"]

if simulationType == 0:
    constraintType = "Euler"
    hessianApprox = "limited-memory"
elif simulationType == 1:
    constraintType = "RK4"
    hessianApprox = "limited-memory"
elif simulationType == 2:
    constraintType = "Euler"
    hessianApprox = "exact"
elif simulationType == 3:
    constraintType = "RK4"
    hessianApprox = "exact"

itersDict = {}

for maxIter in iters:
    if whichFolder == 0:
        timeFolder = "/home/gbehrendt/CLionProjects/Satellite/finalTiming/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
        resultFolder = "/home/gbehrendt/CLionProjects/Satellite/finalResults/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
    elif whichFolder == 1:
        timeFolder = "/home/gbehrendt/CLionProjects/Satellite/neoTiming5/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
        resultFolder = "/home/gbehrendt/CLionProjects/Satellite/neoResults5/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
    elif whichFolder == 2:
        timeFolder = "/home/gbehrendt/CLionProjects/Satellite/neoTiming6/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
        resultFolder = "/home/gbehrendt/CLionProjects/Satellite/neoResults6/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
    masterDict = {}
    
    notConverged = []
    directory = os.fsencode(resultFolder)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            readFile = directory.decode("utf-8") + filename
            # print(readFile)
            with open(readFile, mode='r') as file2:
                # Create a CSV reader with DictReader
                csv_reader = csv.DictReader(file2)
                numRows = 0
                for row in csv_reader:
                    if numRows == 0:
                        trial = float(row["Trial"])
                        converged = row["converged?"]
                    numRows = numRows+1
                if converged != 'yes':
                    notConverged.append(trial)
                # print(trial," ",converged)
    
    # print(maxIter," ",notConverged)
    
    directory = os.fsencode(timeFolder)
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
                scaledConstraintViolation = []
                unscaledConstraintViolation = []
                scaledDualInfeasibility = []
                unscaledDualInfeasibility = []
                scaledNLPError = []
                unscaledNLPError = []
                contents = []
                numFound = 0
                
                read = False
                # displaying the contents of the CSV file
                for lines in csv_reader:
                  contents.append(lines)
                  #print(lines)
    
                  if lines:
                    if 'OverallAlgorithm....................: ' in lines[0]:
                          line = lines[0]
                          res = line.split('OverallAlgorithm....................:      ', 1)
                          if len(res) == 1:
                              res = line.split('OverallAlgorithm....................:     ', 1)
                          if len(res) == 1:
                              res = line.split('OverallAlgorithm....................:    ', 1)
                          new = res[1]
                          new1 = float(new.split(' ',1)[0])
                          timing.append(new1)
                    if 'Constraint violation....:' in lines[0]:
                        # print("constrant violation")
                        line = lines[0]
                        res = line.split('Constraint violation....:   ', 1)
                        temp = res[1].split('    ',1)
                        scaledConstraintViolation.append(float(temp[0]))
                        unscaledConstraintViolation.append(float(temp[1]))
                    if 'Dual infeasibility......:' in lines[0]:
                        # print("dual infeasibility")
                        line = lines[0]
                        res = line.split('Dual infeasibility......:   ', 1)
                        temp = res[1].split('    ',1)
                        scaledDualInfeasibility.append(float(temp[0]))
                        unscaledDualInfeasibility.append(float(temp[1]))
                    if 'Overall NLP error.......:' in lines[0]:
                        # print("dual infeasibility")
                        line = lines[0]
                        res = line.split('Overall NLP error.......:   ', 1)
                        temp = res[1].split('    ',1)
                        scaledNLPError.append(float(temp[0]))
                        unscaledNLPError.append(float(temp[1]))

                        
                        
    
                numLoops = len(timing)
                tt = np.arange(numLoops)
                
                trialDict = {'trial':filename, 'trialNum':trialNum[0], 'timing':timing,
                             'scaledConstraintViolation':scaledConstraintViolation,'unscaledConstraintViolation': unscaledConstraintViolation,
                             'scaledDualInfeasibility':scaledDualInfeasibility,'unscaledDualInfeasibility': unscaledDualInfeasibility,
                             'scaledNLPError':scaledNLPError,'unscaledNLPError': unscaledNLPError, 
                             'numLoops':numLoops, 'tt':tt}
                masterDict[trialNum[0]] = trialDict
    
    # Make list of timing lists
    allTimings = []
    allScaledViolations = []
    allUnscaledViolations = []
    allScaledDualInfeasibility = []
    allUnscaledDualInfeasibility = []
    allScaledNLPError = []
    allUnscaledNLPError = []
    
    for key in masterDict:
        allTimings.append(masterDict[key]['timing'])
        allScaledViolations.append(masterDict[key]['scaledConstraintViolation'])
        allUnscaledViolations.append(masterDict[key]['unscaledConstraintViolation'])
        allScaledDualInfeasibility.append(masterDict[key]['scaledDualInfeasibility'])
        allUnscaledDualInfeasibility.append(masterDict[key]['unscaledDualInfeasibility'])
        allScaledNLPError.append(masterDict[key]['scaledNLPError'])
        allUnscaledNLPError.append(masterDict[key]['unscaledNLPError'])
    
    avgTiming= list(map(get_avg, it.zip_longest(*allTimings)))
    avgScaledViolation= list(map(get_avg, it.zip_longest(*allScaledViolations)))
    avgUnscaledViolation= list(map(get_avg, it.zip_longest(*allUnscaledViolations)))
    avgScaledDualInfeasibility= list(map(get_avg, it.zip_longest(*allScaledDualInfeasibility)))
    avgUnscaledDualInfeasibility= list(map(get_avg, it.zip_longest(*allUnscaledDualInfeasibility)))
    avgScaledNLPError= list(map(get_avg, it.zip_longest(*allScaledNLPError)))
    avgUnscaledNLPError= list(map(get_avg, it.zip_longest(*allUnscaledNLPError)))
    avgTimescale = np.arange(len(avgTiming))
    
    itersDict[maxIter] = {'avgTiming':avgTiming, 'avgScaledViolation':avgScaledViolation, 'avgUnscaledViolation':avgUnscaledViolation, 
                          'avgScaledDualInfeasibility':avgScaledDualInfeasibility, 'avgUnscaledDualInfeasibility':avgUnscaledDualInfeasibility,
                          'avgScaledNLPError':avgScaledNLPError, 'avgUnscaledNLPError':avgUnscaledNLPError,
                          'avgTimescale':avgTimescale}

    
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
ax2.set_yscale('log')
ax2.set_ylabel("IpOpt Algorithm Time $(s)$", fontsize =14)
ax2.set_xlabel("Loop #", fontsize =14)
ax2.set_title(r"Average Time Per Loop", fontsize =14)
ax2.legend(fontsize =8, title="Maximum Iterations",bbox_to_anchor=(1.0, 1.0))
ax2.grid()
plt.show()

fig3, ax3 = plt.subplots()
plt.style.use('default')
for key in itersDict:
    ax3.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgScaledViolation'], label = key)
ax3.set_yscale('log')
ax3.set_xlabel("Loop #", fontsize =14)
ax3.set_title(r"Average Scaled Constraint Violation", fontsize =14)
ax3.legend(fontsize =8, title="Maximum Iterations",bbox_to_anchor=(1.0, 1.0))
ax3.grid()
plt.show()

# fig4, ax4 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     ax4.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgUnscaledViolation'], label = key)
# ax4.set_yscale('log')
# ax4.set_xlabel("Loop #", fontsize =14)
# ax4.set_title(r"Average Unscaled Constraint Violation", fontsize =14)
# ax4.legend(fontsize =12, title="Maximum Iterations")
# ax4.grid()
# plt.show()

# fig5, ax5 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     ax5.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgScaledDualInfeasibility'], label = key)
# ax5.set_yscale('log')
# ax5.set_xlabel("Loop #", fontsize =14)
# ax5.set_title(r"Average Scaled Dual Infeasibility", fontsize =14)
# ax5.legend(fontsize =8, title="Maximum Iterations")
# ax5.grid()
# plt.show()

# fig6, ax6 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     ax6.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgUnscaledDualInfeasibility'], label = key)
# ax6.set_yscale('log')
# ax6.set_xlabel("Loop #", fontsize =14)
# ax6.set_title(r"Average Unscaled Dual Infeasibility", fontsize =14)
# ax6.legend(fontsize =12, title="Maximum Iterations")
# ax6.grid()
# plt.show()

fig7, ax7 = plt.subplots()
plt.style.use('default')
for key in itersDict:
    ax7.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgScaledNLPError'], label = key)
ax7.set_yscale('log')
ax7.set_xlabel("Loop #", fontsize =14)
ax7.set_title(r"Average Scaled NLP Error", fontsize =14)
ax7.legend(loc="lower left", fontsize =8, title="Maximum Iterations")
ax7.grid()
plt.show()

# fig8, ax8 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     ax8.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgUnscaledNLPError'], label = key)
# ax8.set_yscale('log')
# ax8.set_xlabel("Loop #", fontsize =14)
# ax8.set_title(r"Average Unscaled NLP Error", fontsize =14)
# ax8.legend(fontsize =12, title="Maximum Iterations")
# ax8.grid()
# plt.show()

# %%
# toEdit = []
# editedAllTimings = []
        
# for key in masterDict:
#     # print(masterDict[key]['timing'][5:-1])
#     if max(masterDict[key]['timing'][20:-1]) >= 10:
#         # print(key)
#         print(masterDict[key]['trial'])
#         toEdit.append(int(key))
#     else:
#         editedAllTimings.append(masterDict[key]['timing'])
        
        
# newConstraintViolation = []
# newNLPError = []
# overHere = 0

# # print(len(editedAllTimings))

# # for i in range(len(allTimings)):
# #     if i not in toEdit:
# #         print(i,"NO over here" , max(allTimings[i]))
# #         editedAllTimings.append(allTimings[i])
# #         newConstraintViolation.append(allScaledViolations[i])
# #         newNLPError.append(allScaledNLPError[i])
# #         overHere += 1
# #     else:
# #         print(i,"in here", max(allTimings[i]))

# # print(overHere)
# # print(len(editedAllTimings))

# fig60, ax60 = plt.subplots()
# plt.style.use('default')
# for item in editedAllTimings:
#     ax60.plot(np.arange(len(item)), item)
# # ax82.set_yscale('log')
# ax60.set_xlabel("Loop #", fontsize =14)
# ax60.set_title(r"Average Timings $j_{\max} =$" + maxIter , fontsize =14)
# # ax80.legend(fontsize =12, title="Maximum Iterations")
# ax60.grid()
# plt.show()

# fig61, ax61 = plt.subplots()
# plt.style.use('default')
# for item in toEdit:
#     print(item)
#     ax61.plot(np.arange(len(masterDict[str(item)]['timing'])), masterDict[str(item)]['timing'], label = key)


# ax82.set_yscale('log')
# ax60.set_xlabel("Loop #", fontsize =14)
# ax60.set_title(r"Average Timings $j_{\max} =$" + maxIter , fontsize =14)
# # ax80.legend(fontsize =12, title="Maximum Iterations")
# ax60.grid()
# plt.show()

# fig61, ax61 = plt.subplots()
# plt.style.use('default')
# for item in newConstraintViolation:
#     ax61.plot(np.arange(len(item)), item, label = key)
# ax61.set_yscale('log')
# ax61.set_xlabel("Loop #", fontsize =14)
# ax61.set_title(r"Scaled Constraint Violation $j_{\max} =$" + maxIter , fontsize =14)
# # ax80.legend(fontsize =12, title="Maximum Iterations")
# ax61.grid()
# plt.show()

# fig62, ax62 = plt.subplots()
# plt.style.use('default')
# for item in newNLPError:
#     ax62.plot(np.arange(len(item)), item, label = key)
# ax62.set_yscale('log')
# ax62.set_xlabel("Loop #", fontsize =14)
# ax62.set_title(r"Scaled NLP Error $j_{\max} =$" + maxIter , fontsize =14)
# # ax80.legend(fontsize =12, title="Maximum Iterations")
# ax62.grid()
# plt.show()




# %%
# fig80, ax80 = plt.subplots()
# plt.style.use('default')
# for item in allScaledNLPError:
#     ax80.plot(np.arange(len(item)), item, label = key)
# ax80.set_yscale('log')
# ax80.set_xlabel("Loop #", fontsize =14)
# ax80.set_title(r"Average Scaled NLP Error $j_{\max} =$" + maxIter , fontsize =14)
# # ax80.legend(fontsize =12, title="Maximum Iterations")
# ax80.grid()
# plt.show()

# fig81, ax81 = plt.subplots()
# plt.style.use('default')
# for item in allScaledViolations:
#     ax81.plot(np.arange(len(item)), item, label = key)
# ax81.set_yscale('log')
# ax81.set_xlabel("Loop #", fontsize =14)
# ax81.set_title(r"Average Scaled Constraint Violation $j_{\max} =$" + maxIter , fontsize =14)
# # ax80.legend(fontsize =12, title="Maximum Iterations")
# ax81.grid()
# plt.show()

# fig83, ax83 = plt.subplots()
# plt.style.use('default')
# for item in allScaledDualInfeasibility:
#     ax83.plot(np.arange(len(item)), item, label = key)
# ax83.set_yscale('log')
# ax83.set_xlabel("Loop #", fontsize =14)
# ax83.set_title(r"Average Scaled Dual Infeasibility $j_{\max} =$" + maxIter , fontsize =14)
# # ax80.legend(fontsize =12, title="Maximum Iterations")
# ax83.grid()
# plt.show()

# fig82, ax82 = plt.subplots()
# plt.style.use('default')
# for item in allTimings:
#     ax82.plot(np.arange(len(item)), item, label = key)
# # ax82.set_yscale('log')
# ax82.set_xlabel("Loop #", fontsize =14)
# ax82.set_title(r"All Timings $j_{\max} =$" + maxIter , fontsize =14)
# # ax80.legend(fontsize =12, title="Maximum Iterations")
# ax82.grid()
# plt.show()














