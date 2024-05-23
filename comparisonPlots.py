import csv
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import linalg as LA
from itertools import zip_longest
import re
import itertools as it
from matplotlib import ticker
import bisect

# Define a function 'get_avg' that calculates the average of a list, ignoring 'None' values.
def get_avg(x):
    # Remove 'None' values from the list 'x'.
    x = [i for i in x if i is not None]
    # Calculate the average of the remaining values.
    return sum(x, 0.0) / len(x)

whichFolder = 2
savePlots = False
samplingTime = "10"
constraintType = "Euler"

# iters = ["5","6","7","8","9","10","15","50","100","200","300","400","500","600","700","900","1000","2000","10000"] # All iterations ran
# iters = ["5","6","7","8","9","10","15","50","100","101","150","200","300","400","500","600","700","900","1000","10000"] # All iterations ran
iters = ["5","6","7","8","9","10","15","50","100","125","150","175","200","300","400","500","1000","10000"] # zResults
# iters = ["5","6","7","8","9","10","15","50","100","125","150","175","200","1000","10000"] # zResults



itersTimingDict = {}
itersResultDict = {}

for maxIter in iters:
    print("%%%%%%%%%%",maxIter,"%%%%%%%%%%")
    if whichFolder == 0:
        timeFolder = "/home/gbehrendt/CLionProjects/Satellite/finalTimingExactNoCons3/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
        resultFolder = "/home/gbehrendt/CLionProjects/Satellite/finalResultsExactNoCons3/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
    elif whichFolder == 1:
        timeFolder = "/home/gbehrendt/CLionProjects/Satellite/zTiming/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
        resultFolder = "/home/gbehrendt/CLionProjects/Satellite/zResults/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
    elif whichFolder == 2:
        timeFolder = "/home/gbehrendt/CLionProjects/Satellite/Timing/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
        resultFolder = "/home/gbehrendt/CLionProjects/Satellite/Results/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
    elif whichFolder == 3:
        timeFolder = "/home/gbehrendt/CLionProjects/Satellite/newQuatTiming/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
        resultFolder = "/home/gbehrendt/CLionProjects/Satellite/newQuatResults/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
    elif whichFolder == 4:
        timeFolder = "/home/gbehrendt/CLionProjects/Satellite/newQuatTiming4/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
        resultFolder = "/home/gbehrendt/CLionProjects/Satellite/newQuatResults4/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"


    
    masterResultDict = {}
    masterTimingDict = {}
    numConverged = 0
    numNotConverged = 0
    
    directory = os.fsencode(resultFolder)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"): 
            readFile = directory.decode("utf-8") + filename
            #print(readFile)
            
            with open(readFile, mode='r') as file2:
                # Create a CSV reader with DictReader
                csv_reader = csv.DictReader(file2)
             
                # Iterate through each row in the CSV file
                numStates = 13
                numRows = 0
                x0 = []
                x = []
                y = []
                z = []
                dx = []
                dy = []
                dz = []
                sq = []
                v1 = []
                v2 = []
                v3 = []
                dw1 = []
                dw2 = []
                dw3 = []
                thrust1 = []
                thrust2 = []        # print(maxIter,"  ",key)
                thrust3 =[]
                tau1 = []
                tau2 =[]
                tau3 = []
        
                for row in csv_reader:
                    if numRows == 0:
                        maxIter = int(row["Maximum Iterations"])
                        ts = float(row["ts"])
                        N = int(row["N"])
                        numLoops = int(row["MPC Loops"])
                        posCost = float(row["posCost"])
                        velCost = float(row["velCost"])
                        quatCost = float(row["quatCost"])
                        angularCost = float(row["angualarCost"])
                        thrustCost = float(row["thrustCost"])
                        torqueCost = float(row["torqueCost"])
                        thrustMax = float(row["thrustMax"])
                        torqueMax = float(row["torqueMax"])
                        trial = float(row["Trial"])
                        infNorm = float(row["infNorm"])
                        converged = row["converged?"]
                    
                    if numRows < numStates:
                        x0.append(float(row["x0"]))
                        
                    
                    # read in state trajectories
                    x.append(float(row["x (km)"]))
                    y.append(float(row["y (km)"]))
                    z.append(float(row["z (km)"]))
                    dx.append(float(row["xdot (km/s)"]))
                    dy.append(float(row["ydot (km/s)"]))
                    dz.append(float(row["zdot (km/s)"]))
                    sq.append(float(row["sq"]))
                    v1.append(float(row["v1"]))
                    v2.append(float(row["v2"]))
                    v3.append(float(row["v3"]))
                    dw1.append(float(row["dw1 (rad/s)"]))
                    dw2.append(float(row["dw2 (rad/s)"]))
                    dw3.append(float(row["dw3 (rad/s)"]))
                    
                    # read in control
                    thrust1.append(float(row["thrust1 (N)"]))
                    thrust2.append(float(row["thrust2 (N)"]))
                    thrust3.append(float(row["thrust3 (N)"]))
                    tau1.append(float(row["tau1 (rad/s^2)"]))
                    tau2.append(float(row["tau2 (rad/s^2)"]))
                    tau3.append(float(row["tau3 (rad/s^2)"]))
                    
                    
                    numRows = numRows+1
    
        
            # store time axis values
            tt = np.arange(numLoops)*ts #np.linspace(0,numLoops*ts,numLoops)
            endTime = numLoops*ts
            
            # Calculate Objective Function Value
            Q = np.diag([posCost, posCost, posCost, velCost, velCost, velCost,
                         quatCost, quatCost, quatCost, quatCost, angularCost, angularCost, angularCost])
            R = np.diag([thrustCost, thrustCost, thrustCost, torqueCost, torqueCost, torqueCost])
            
            # Compute error
            xd = np.array([0,0,0,0,0,0,1,0,0,0,0,0,0])
            errorSt = []
            
            errorPos = []
            errorVel = []
            errorTrans = []
            errorQuat = []
            errorAngVel = []
            errorRot = []
            
            errorCon = []
            errorThrust = []
            errorTau = []
            
            totError = []
            cost = []
            #infNorm = []
            for i in range(numLoops):
                curSt = np.array([x[i], y[i], z[i], dx[i], dy[i], dz[i], sq[i], v1[i], v2[i], v3[i], dw1[i], dw2[i], dw3[i]])
                curCon = np.array([ thrust1[i], thrust2[i], thrust3[i], tau1[i], tau2[i], tau3[i] ])
                errorSt.append( LA.norm(curSt - xd) )
                errorCon.append(LA.norm(curCon))
                
                errorThrust.append(LA.norm(curCon[0:3]))
                errorTau.append(LA.norm(curCon[3:6]))
                
                errorPos.append(LA.norm(curSt[0:3]))
                errorVel.append(LA.norm(curSt[3:6]))
                errorTrans.append(LA.norm(curSt[0:7]))
                
                errorQuat.append(LA.norm(curSt[6:10] - xd[6:10]))
                errorAngVel.append(LA.norm(curSt[10:13]))
                errorRot.append(LA.norm(curSt[6:13]) - xd[6:13])
                
                totError.append(LA.norm(np.concatenate(((curSt-xd),curCon))))
                cost.append((curSt - xd).T @ Q @ (curSt - xd) + curCon.T @ R @ curCon)
            
            totCost = sum(cost)
            
            # Store data in a dictionary
            trialResultDict = {'x':x,'y':y,'z':z,
                        'dx':dx,'dy':dy,'dz':dz,
                        'sq':sq,'v1':v1,'v2':v2,'v3':v3,
                        'dw1':dw1,'dw2':dw2,'dw3':dw3,
                        'thrust1':thrust1,'thrust2':thrust2,'thrust3':thrust3,
                        'tau1':tau1,'tau2':tau2,'tau3':tau3,
                        'maxIter':maxIter,'ts':ts, 'N':N, 'numLoops':numLoops,
                        'posCost':posCost, 'velCost':velCost, 'quatCost':quatCost, 'angularCost':angularCost,
                        'thrustCost':thrustCost, 'torqueCost':torqueCost, 'thrustMax':thrustMax, 'torqueMax':torqueMax,
                        'tt':tt, 'endTime':endTime, 'errorSt':errorSt, 'errorCon':errorCon,
                        'errorPos':errorPos, 'errorVel':errorVel, 'errorTrans':errorTrans, 
                        'errorQuat':errorQuat,'errorAngVel':errorAngVel, 'errorRot':errorRot,
                        'errorThrust':errorThrust, 'errorTau':errorTau,
                        'cost':cost, 'totError':totError,'totCost':totCost, 'trial':trial,
                        'infNorm':infNorm, 'x0':x0
                        }
            if(converged == "yes"):
                masterResultDict[trial] = trialResultDict
                numConverged += 1
            else:
                masterResultDict[trial] = trialResultDict
                numNotConverged += 1
            continue
        else:
            continue
        
    # Compute average cost and error
    allErrorSt = []
    allErrorCon = []
    allError = []
    allCost = []

    allErrorPos = []
    allErrorVel = []
    allErrorTrans = []
    
    allErrorQuat = []
    allErrorAngVel = []
    allErrorRot = []
    
    allErrorThrust = []
    allErrorTau = []
    
    
    
    for key in masterResultDict:
        allErrorSt.append(masterResultDict[key]['errorSt'])
        allErrorCon.append(masterResultDict[key]['errorCon'])
        allError.append(masterResultDict[key]['totError'])
        allCost.append(masterResultDict[key]['cost'])
        
        allErrorPos.append(masterResultDict[key]['errorPos'])
        allErrorVel.append(masterResultDict[key]['errorVel'])
        allErrorTrans.append(masterResultDict[key]['errorTrans'])
        
        allErrorQuat.append(masterResultDict[key]['errorQuat'])
        allErrorAngVel.append(masterResultDict[key]['errorAngVel'])
        allErrorRot.append(masterResultDict[key]['errorRot'])
        
        allErrorThrust.append(masterResultDict[key]['errorThrust'])
        allErrorTau.append(masterResultDict[key]['errorTau'])
    
    avgErrorCon = list(map(get_avg, it.zip_longest(*allErrorCon)))
    avgErrorSt = list(map(get_avg, it.zip_longest(*allErrorSt)))
    avgError = list(map(get_avg, it.zip_longest(*allError)))
    avgCost = list(map(get_avg, it.zip_longest(*allCost)))
    
    avgErrorPos = list(map(get_avg, it.zip_longest(*allErrorPos)))
    avgErrorVel = list(map(get_avg, it.zip_longest(*allErrorVel)))
    avgErrorTrans = list(map(get_avg, it.zip_longest(*allErrorTrans)))
    
    avgErrorQuat = list(map(get_avg, it.zip_longest(*allErrorQuat)))
    avgErrorAngVel = list(map(get_avg, it.zip_longest(*allErrorAngVel)))
    avgErrorRot = list(map(get_avg, it.zip_longest(*allErrorRot)))
    
    avgErrorThrust = list(map(get_avg, it.zip_longest(*allErrorThrust)))
    avgErrorTau = list(map(get_avg, it.zip_longest(*allErrorTau)))
    
    totAvgCost = sum(avgCost)
    
    
    
    
    avgTimescale = np.arange(len(avgErrorSt))*ts
    
    
    itersResultDict[maxIter] = {'avgErrorSt':avgErrorSt, 'avgErrorCon':avgErrorCon, 'avgError':avgError, 'avgCost':avgCost,
                          'avgErrorPos':avgErrorPos, 'avgErrorVel':avgErrorVel, 'avgErrorTrans':avgErrorTrans,
                          'avgErrorQuat':avgErrorQuat, 'avgErrorAngVel':avgErrorAngVel, 'avgErrorRot':avgErrorRot,
                          'avgErrorThrust':avgErrorThrust, 'avgErrorTau':avgErrorTau,
                          'totAvgCost':totAvgCost,'avgTimescale':avgTimescale}
        
    # numTrials = numConverged + numNotConverged
    # print("# Converged:", numConverged)
    # print("# Not Converged:", numNotConverged)
    # print("Total Run:", numTrials)
    
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
                          # print(line)
                          # print(res)
                          new = res[1]
                          new1 = float(new.split(' ',1)[0])
                          timing.append(new1)
                    if 'Constraint violation....:' in lines[0]:
                        # print("constrant violation")
                        line = lines[0]
                        res = line.split('Constraint violation....:   ', 1)
                        if len(res) == 1:
                            res = line.split('Constraint violation....:  ', 1)
                        # print(line,"  ",res)
                        # print(res)
                        temp = res[1].split('    ',1)
                        if len(temp) == 1:
                            temp = res[1].split('   ',1)
                        # print(temp)
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
                        line = lines[0]
                        res = line.split('Overall NLP error.......:   ', 1)
                        if len(res) == 1:
                            res = line.split('Overall NLP error.......:  ', 1)
                        # print(res)
                        temp = res[1].split('    ',1)
                        if len(temp) == 1:
                            temp = res[1].split('   ',1)
                        # print(temp)
                        scaledNLPError.append(float(temp[0]))
                        unscaledNLPError.append(float(temp[1]))

                        
                        
    
                numLoops = len(timing)
                tt = np.arange(numLoops)
                maxTime = max(timing)
                
                trialTimingDict = {'trial':filename, 'trialNum':trialNum[0], 'timing':timing,
                             'scaledConstraintViolation':scaledConstraintViolation,'unscaledConstraintViolation': unscaledConstraintViolation,
                             'scaledDualInfeasibility':scaledDualInfeasibility,'unscaledDualInfeasibility': unscaledDualInfeasibility,
                             'scaledNLPError':scaledNLPError,'unscaledNLPError': unscaledNLPError, 
                             'numLoops':numLoops, 'tt':tt, 'maxTime':maxTime }
                masterTimingDict[trialNum[0]] = trialTimingDict
    
    # Make list of timing lists
    allTimings = []
    allScaledViolations = []
    allUnscaledViolations = []
    allScaledDualInfeasibility = []
    allUnscaledDualInfeasibility = []
    allScaledNLPError = []
    allUnscaledNLPError = []
    maxLoopTime = 0
    
    for key in masterTimingDict:
        allTimings.append(masterTimingDict[key]['timing'])
        allScaledViolations.append(masterTimingDict[key]['scaledConstraintViolation'])
        allUnscaledViolations.append(masterTimingDict[key]['unscaledConstraintViolation'])
        allScaledDualInfeasibility.append(masterTimingDict[key]['scaledDualInfeasibility'])
        allUnscaledDualInfeasibility.append(masterTimingDict[key]['unscaledDualInfeasibility'])
        allScaledNLPError.append(masterTimingDict[key]['scaledNLPError'])
        allUnscaledNLPError.append(masterTimingDict[key]['unscaledNLPError'])
        
        if masterTimingDict[key]['maxTime'] > maxLoopTime:
            maxLoopTime = masterTimingDict[key]['maxTime']
    
    avgTiming= list(map(get_avg, it.zip_longest(*allTimings)))
    avgScaledViolation= list(map(get_avg, it.zip_longest(*allScaledViolations)))
    avgUnscaledViolation= list(map(get_avg, it.zip_longest(*allUnscaledViolations)))
    avgScaledDualInfeasibility= list(map(get_avg, it.zip_longest(*allScaledDualInfeasibility)))
    avgUnscaledDualInfeasibility= list(map(get_avg, it.zip_longest(*allUnscaledDualInfeasibility)))
    avgScaledNLPError= list(map(get_avg, it.zip_longest(*allScaledNLPError)))
    avgUnscaledNLPError= list(map(get_avg, it.zip_longest(*allUnscaledNLPError)))
    avgTimescale = np.arange(len(avgTiming))
    
    # print(avgTiming)
    avgTimePerLoop = sum(avgTiming) / len(avgTiming)
    avgViolationPerLoop = sum(avgScaledViolation) / len(avgScaledViolation)
    
    totAvgTiming = sum(avgTiming)
    totAvgViolation = sum(avgScaledViolation)
    totAvgNLPError = sum(avgScaledNLPError)

    binSize = 0.025
    maxTime = 2.0
    binLimits = np.arange(0,maxTime,binSize)
    sumBins = np.zeros(int(maxTime/binSize))
    # print(key1,sumBins)
    for key in masterTimingDict:
        masterTimingDict[key]["bins"] = np.zeros(int(20/binSize))
        # print("******* ", key," *********")
        for ele in masterTimingDict[key]["timing"]:
            i = bisect.bisect(binLimits, ele)
            # print(ele,i,binLimits[i-1],binLimits[i])
            # print(masterTimingDict[key]["bins"])
            masterTimingDict[key]["bins"][i] += 1
        
        for i in range(len(sumBins)):
            sumBins[i] = sumBins[i] + masterTimingDict[key]["bins"][i]
    # print(sumBins)
    # print(key1,itersTimingDict[key1]["sumBins"])
        
    
    itersTimingDict[maxIter] = {'avgTiming':avgTiming, 'avgScaledViolation':avgScaledViolation, 'avgUnscaledViolation':avgUnscaledViolation, 
                          'avgScaledDualInfeasibility':avgScaledDualInfeasibility, 'avgUnscaledDualInfeasibility':avgUnscaledDualInfeasibility,
                          'avgScaledNLPError':avgScaledNLPError, 'avgUnscaledNLPError':avgUnscaledNLPError,
                          'avgTimescale':avgTimescale,'totAvgViolation':totAvgViolation, 'totAvgNLPError':totAvgNLPError,
                          'totAvgTiming':totAvgTiming, 'avgTimePerLoop':avgTimePerLoop, 'avgViolationPerLoop':avgViolationPerLoop,
                          'sumBins':sumBins, 'maxLoopTime':maxLoopTime }
    
# for key in itersTimingDict:
#     print("******* ", key," *********")
#     print(itersTimingDict[key]['sumBins'])
 # %% Timing Plots 
numBars = len(itersResultDict)
colormap = plt.cm.get_cmap('jet', numBars)
myColor =[colormap(i) for i in range(numBars)]

colormap = plt.cm.get_cmap('jet', numBars)
myColor2 =[colormap(i) for i in range(numBars)]

myColor = [ "1FCC6D","4B8BEB", "D64215", "F58201", "0C7500", "820099", "00D6DB", "A87CD6", "4502EB", "EB1189", "9E8001" ]

myColor = ["01CC58", "0075EB", "D64215", "F58201", "1D7500", "820099", "01B5DB", "FF02EB","00D3B4", "F5001E",
           "F0C055", "9E8001", "A87CD6", "BB4E8F", "B0C700", "DB8E73", "64FAC4", "001CBD","EB1189"]

for i in range(len(myColor)):
    h = myColor[i]
    myColor[i] = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
    # print(myColor[i])
myColor2 = myColor

i = 0
fig1, ax1 = plt.subplots()
plt.style.use('default')
for key in itersTimingDict:
    lbl = key
    if lbl == 10000 or lbl == '10000':
        lbl = 'Optimal'
    ax1.plot(itersTimingDict[key]['avgTimescale'], itersTimingDict[key]['avgTiming'], color = myColor2[i], label = lbl)
    i+=1
ax1.set_yscale('log')
ax1.set_ylabel("IpOpt Algorithm Time $(s)$", fontsize =14)
ax1.set_xlabel("MPC Loop #", fontsize =14)
ax1.set_ylim([0,1])
# ax2.set_title(r"Average Time Per Loop", fontsize =14)
leg = ax1.legend(fontsize =10,loc = "upper right", ncol = 5)
leg.set_title(r'$j_{\max}$',prop={'size':14})
ax1.grid()
if savePlots == True:
    plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgTiming.eps', format = 'eps', bbox_inches='tight')
plt.show()

i = 0
fig9, ax9 = plt.subplots()
plt.style.use('default')
for key in itersTimingDict:
    lbl = key
    if lbl == 10000 or lbl == '10000':
        lbl = 'Optimal'
    ax9.bar(str(lbl), itersTimingDict[key]['maxLoopTime'], color = myColor[i], edgecolor ='black')
    i += 1
ax9.set_ylabel(r'Maximum Loop Time $(s)$', fontsize =14)
ax9.set_xlabel(r'$j_{\max}$', fontsize = 14)
# ax11.set_yscale('log')
ax9.grid(axis='y')
ax9.set_axisbelow(True)
fig9.autofmt_xdate()
if savePlots == True:
    plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/maxLoopTime.eps', format = 'eps', bbox_inches='tight')
plt.show()


i = 0
fig2, ax2 = plt.subplots()
plt.style.use('default')
for key in itersTimingDict:
    lbl = key
    if lbl == 10000 or lbl == '10000':
        lbl = 'Optimal'
    ax2.bar(str(lbl), itersTimingDict[key]['avgTimePerLoop'], color = myColor[i], edgecolor ='black')
    i += 1
ax2.set_ylabel(r'Average Time Per Loop $(s)$', fontsize =14)
ax2.set_xlabel(r'$j_{\max}$', fontsize = 14)
# ax11.set_yscale('log')
ax2.grid(axis='y')
ax2.set_axisbelow(True)
fig2.autofmt_xdate()
if savePlots == True:
    plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgTimePerLoop.eps', format = 'eps', bbox_inches='tight')
plt.show()

 # %% Constraint Violation Plots
i = 0
fig3, ax3 = plt.subplots()
plt.style.use('default')
for key in itersTimingDict:
    lbl = key
    if lbl == 10000 or lbl == '10000':
        lbl = 'Optimal'
    ax3.plot(itersTimingDict[key]['avgTimescale'], itersTimingDict[key]['avgScaledViolation'], color = myColor2[i], label = lbl)
    i+=1
ax3.set_yscale('log')
ax3.set_xlabel("MPC Loop #", fontsize =14)
ax3.set_ylabel(r"Average Scaled Constraint Violation", fontsize =14)
ax3.set_ylim([0,1000000000])
leg = ax3.legend(fontsize =9,loc = "upper right", ncol = 4)
leg.set_title(r'$j_{\max}$',prop={'size':12})
ax3.grid()
if savePlots == True:
    plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgScaledViolation.eps', format = 'eps', bbox_inches='tight')
plt.show()


i = 0
fig4, ax4 = plt.subplots()
plt.style.use('default')
for key in itersTimingDict:
    lbl = key
    if lbl == 10000 or lbl == '10000':
        lbl = 'Optimal'
    ax4.bar(str(lbl), itersTimingDict[key]['totAvgViolation'], color = myColor[i], edgecolor ='black')
    i+=1
ax4.set_ylabel(r'Total Average Constraint Violation', fontsize =14)
ax4.set_xlabel(r'$j_{\max}$', fontsize = 14)
ax4.set_yscale('log')
ax4.grid(axis='y')
ax4.set_axisbelow(True)
fig4.autofmt_xdate()
if savePlots == True:
    plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/totAvgViolation.eps', format = 'eps', bbox_inches='tight')
plt.show()


 # %% Cost Plots
i = 0
fig32, ax32 = plt.subplots()
plt.style.use('default')
for key in itersResultDict:
    lbl = key
    if lbl == 10000 or lbl == '10000':
        lbl = 'Optimal'
    ax32.plot(itersResultDict[key]['avgTimescale'], itersResultDict[key]['avgCost'], color = myColor2[i], label = lbl)
    i+=1
ax32.set_yscale('log')
ax32.set_ylabel(r'$(x(k) - x_d)^T Q (x(k) - x_d) + u(k)^T R u(k)$', fontsize =14)
ax32.set_xlabel("Time $(s)$", fontsize =14)
# ax32.set_title(r"Average Cost", fontsize =14)
leg = ax32.legend(fontsize =10,loc = "lower left", ncol = 3)
leg.set_title(r'$j_{\max}$',prop={'size':12})
ax32.grid()
if savePlots == True:
    plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgCost.eps', format = 'eps', bbox_inches='tight')
plt.show()

i = 0
fig36, ax36 = plt.subplots()
values = []
for key in itersResultDict:
    lbl = key
    if lbl == 10000 or lbl == '10000':
        lbl = 'Optimal'
    values.append(itersResultDict[key]['totAvgCost'])
    ax36.bar(str(lbl), itersResultDict[key]['totAvgCost'], color = myColor[i],edgecolor ='black')
    i+=1
yfmt = ticker.ScalarFormatter(useMathText=True)
yfmt.set_powerlimits((3, 4))
ax36.set_ylabel(r'Total Average Cost', fontsize =14)
ax36.set_xlabel(r'$j_{\max}$', fontsize = 14)
# ax36.set_yscale('log')
# ax36.yaxis.set_major_formatter(yfmt)
# ax36.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
# ax36.get_yaxis().get_offset_text().set_visible(False)
# ax_max = max(ax36.get_yticks())
# exponent_axis = np.floor(np.log10(ax_max)).astype(int)
# ax36.annotate(r'$\times$10$^{12}$',
#              xy=(.01, .96), xycoords='axes fraction')
# ax36.set_ylim([8e12,9.3e12])
ax36.grid(axis='y')
ax36.set_axisbelow(True)
fig36.autofmt_xdate()
if savePlots == True:
    plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/totAvgCost.eps', format = 'eps', bbox_inches='tight')
plt.show()
plt.show()



values = []
# for key in itersResultDict:
#     values.append(itersResultDict[key]['totAvgCost'])
    # print(values)
for key in itersResultDict:
    itersResultDict[key]['relTotAvgCost'] = itersResultDict[key]['totAvgCost'] / itersResultDict[10000]['totAvgCost']
    # print(itersResultDict[key]['relTotAvgCost'])
    # print(values)


# fig5, ax5 = plt.subplots()
# plt.style.use('default')
# for key in itersResultDict:
#     lbl = key
#     if lbl == '10000' or lbl == 10000:
#         lbl = 'Optimal'
#     ax5.bar(str(lbl), itersResultDict[key]['relTotAvgCost'])
# ax5.set_ylabel(r'Relative Total Average Cost', fontsize =14)
# ax5.set_xlabel(r'$j_{\max}$', fontsize = 14)
# ax5.set_yscale('log')
# ax5.grid(axis='y')
# ax5.set_axisbelow(True)
# if savePlots == True:
#     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/relTotAvgCost.eps', format = 'eps', bbox_inches='tight')
# plt.show()



#%%
### Putting it all together ###
# savePlots = True
timePerLoopList = []
totAvgViolationList = []
relTotCostList = []
labels = []

# get plotting values
for key in itersResultDict:
    lbl = key
    if lbl == '10000' or lbl == 10000:
        lbl = 'Optimal'
    labels.append(lbl)
    timePerLoopList.append(itersTimingDict[key]['avgTimePerLoop'])
    totAvgViolationList.append(itersTimingDict[key]['totAvgViolation'])
    relTotCostList.append(itersResultDict[key]['relTotAvgCost'])

# Set position of bar on X axis
barWidth = 0.25
br1 = np.arange(len(timePerLoopList))
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 

fig7, ax7 = plt.subplots()
plt.style.use('default')
ax7.bar(br1, timePerLoopList, color ='lightcoral', width = barWidth, 
        edgecolor ='black', label =r'Time Per Loop $(s)$') 
ax7.bar(br2, totAvgViolationList, color ='lightgreen', width = barWidth, 
        edgecolor ='black', label ='Average Constraint Violation') 
ax7.bar(br3, relTotCostList, color ='gold', width = barWidth, 
        edgecolor ='black', label ='Relative Total Average Cost') 
# ax7.set_yscale('log')
ax7.grid(axis='y')
ax7.set_axisbelow(True)
ax7.set_xlabel(r'$j_{\max}$', fontsize = 15) 
ax7.set_xticks([r + barWidth for r in range(len(timePerLoopList))], 
        labels)
ax7.set_ylim([0,1.5])
plt.legend(loc="upper right")
fig7.autofmt_xdate()
if savePlots == True:
    plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/comparison.eps', format = 'eps', bbox_inches='tight')
plt.show() 


#%%

# binSize = 0.25
# binLimits = np.arange(0,20,binSize)

# for key1 in itersTimingDict:
#     sumBins = np.zeros(int(20/binSize))
#     print(key1,sumBins)
#     for key in masterTimingDict:
#         masterTimingDict[key]["bins"] = np.zeros(int(20/binSize))
#         # print("******* ", key," *********")
#         for ele in masterTimingDict[key]["timing"]:
#             i = bisect.bisect(binLimits, ele)
#             # print(ele,i,binLimits[i-1],binLimits[i])
#             # print(masterTimingDict[key]["bins"])
#             masterTimingDict[key]["bins"][i] += 1
        
#         for i in range(len(sumBins)):
#             sumBins[i] = sumBins[i] + masterTimingDict[key]["bins"][i]
#     itersTimingDict[key1]["sumBins"] = sumBins
#     print(sumBins)
#     # print(key1,itersTimingDict[key1]["sumBins"])
    
    
    
i = 0
# binLabels = np.arange(0,)


fig8, ax8 = plt.subplots()
plt.style.use('default')
for key in itersTimingDict:
    lbl = str(key)
    if lbl == 10000 or lbl == '10000':
        lbl = 'Optimal'
    # ax8.bar(range(len(binLimits)), itersTimingDict[key]['sumBins'], color = myColor[i],label=lbl)
    ax8.bar(binLimits, itersTimingDict[key]['sumBins'], color = myColor[i],label=lbl, width=binSize - (binSize/3))
    i+=1
ax8.set_ylabel(r'Count', fontsize =14)
ax8.set_xlabel(r'Solution Time $(s)$', fontsize = 14)
ax8.set_yscale('log')
# ax8.set_xscale('log')
ax8.grid(axis='y')
ax8.set_axisbelow(True)
plt.legend(ncol=2)
if savePlots == True:
    plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/timingBins.eps', format = 'eps', bbox_inches='tight')
plt.show()

















