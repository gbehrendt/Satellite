import csv
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import linalg as LA
from itertools import zip_longest
import itertools as it

# Define a function 'get_avg' that calculates the average of a list, ignoring 'None' values.
def get_avg(x):
    # Remove 'None' values from the list 'x'.
    x = [i for i in x if i is not None]
    # Calculate the average of the remaining values.
    return sum(x, 0.0) / len(x)

whichFolder = 0
simulationType = 0

samplingTime = "10"
iters = ["4","5","10","20","50","100"]
iters = ["5","6","7","8","9","10","1000"]



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

notWorking = []

itersDict = {}
for maxIter in iters:
    print("%%%%%%%%%%",maxIter,"%%%%%%%%%%")
    if whichFolder == 0:
        readFolder = "/home/gbehrendt/CLionProjects/Satellite/Results/" + constraintType + "/" + hessianApprox + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
    elif whichFolder == 1:
        readFolder = "/home/gbehrendt/CLionProjects/untitled/parallel/Results3/" + constraintType + "/" + hessianApprox + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
    

    numConverged = 0
    numNotConverged = 0
    masterDict = {}
    
    directory = os.fsencode(readFolder)
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
                thrust2 = []
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
            errorCon = []
            totError = []
            cost = []
            #infNorm = []
            for i in range(numLoops):
                curSt = np.array([x[i], y[i], z[i], dx[i], dy[i], dz[i], sq[i], v1[i], v2[i], v3[i], dw1[i], dw2[i], dw3[i]])
                curCon = np.array([ thrust1[i], thrust2[i], thrust3[i], tau1[i], tau2[i], tau3[i] ])
                errorSt.append( LA.norm(curSt - xd) )
                errorCon.append(LA.norm(curCon))
                totError.append(LA.norm(np.concatenate(((curSt-xd),curCon))))
                cost.append((curSt - xd).T @ Q @ (curSt - xd) + curCon.T @ R @ curCon)
            
            totCost = sum(cost)
            
            # Store data in a dictionary
            trialDict = {'x':x,'y':y,'z':z,
                        'dx':dx,'dy':dy,'dz':dz,
                        'sq':sq,'v1':v1,'v2':v2,'v3':v3,
                        'dw1':dw1,'dw2':dw2,'dw3':dw3,
                        'thrust1':thrust1,'thrust2':thrust2,'thrust3':thrust3,
                        'tau1':tau1,'tau2':tau2,'tau3':tau3,
                        'maxIter':maxIter,'ts':ts, 'N':N, 'numLoops':numLoops,
                        'posCost':posCost, 'velCost':velCost, 'quatCost':quatCost, 'angularCost':angularCost,
                        'thrustCost':thrustCost, 'torqueCost':torqueCost, 'thrustMax':thrustMax, 'torqueMax':torqueMax,
                        'tt':tt, 'endTime':endTime, 'errorSt':errorSt, 'errorCon':errorCon,'cost':cost, 'totError':totError,'totCost':totCost, 'trial':trial,
                        'infNorm':infNorm
                        }
            if(converged == "yes"):
                masterDict[trial] = trialDict
                numConverged += 1
            else:
                # masterDict[trial] = trialDict
                numNotConverged += 1
                notWorking.append(trial)
                # print(trial)
            continue
        else:
            continue
        
    notWorking.sort()
    numTrials = numConverged + numNotConverged
    print("# Converged:", numConverged)
    print("# Not Converged:", numNotConverged)
    print("Total Run:", numTrials)
    
    # Compute average cost and error
    allErrorSt = []
    allErrorCon = []
    allError = []
    allCost = []
    for key in masterDict:
        allErrorSt.append(masterDict[key]['errorSt'])
        allErrorCon.append(masterDict[key]['errorCon'])
        allError.append(masterDict[key]['totError'])
        allCost.append(masterDict[key]['cost'])
    
    avgErrorCon = list(map(get_avg, it.zip_longest(*allErrorCon)))
    avgErrorSt = list(map(get_avg, it.zip_longest(*allErrorSt)))
    avgError = list(map(get_avg, it.zip_longest(*allError)))
    avgCost = list(map(get_avg, it.zip_longest(*allCost)))
    avgTimescale = np.arange(len(avgErrorSt))*ts
    
    
    itersDict[maxIter] = {'avgErrorSt':avgErrorSt, 'avgErrorCon':avgErrorCon, 'avgError':avgError, 'avgCost':avgCost, 'avgTimescale':avgTimescale}
    
    
    
    
    ######################### Position Plots #################################
    # x-Position plot
    fig10, ax10 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax10.plot(masterDict[key]['tt'], masterDict[key]['x'], label = key)
    ax10.set_xlabel("Time $(s)$", fontsize =14)
    ax10.set_title(r"x-position $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax10.grid()
    plt.show()
    
    # y-Position plot
    fig11, ax11 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax11.plot(masterDict[key]['tt'], masterDict[key]['y'], label = key)
    ax11.set_xlabel("Time $(s)$", fontsize =14)
    ax11.set_title(r"y-position $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax11.grid()
    plt.show()
    
    # z-Position plot
    fig12, ax12 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax12.plot(masterDict[key]['tt'], masterDict[key]['z'], label = key)
    ax12.set_xlabel("Time $(s)$", fontsize =14)
    ax12.set_title(r"z-position $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax12.grid()
    plt.show()
    
    ######################### Velocity Plots #########################
    # x-Velocity plot
    fig13, ax13 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax13.plot(masterDict[key]['tt'], masterDict[key]['dx'], label = key)
    ax13.set_xlabel("Time $(s)$", fontsize =14)
    ax13.set_title(r"dx $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax13.grid()
    plt.show()
    
    # y-Velocity plot
    fig14, ax14 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax14.plot(masterDict[key]['tt'], masterDict[key]['dy'], label = key)
    ax14.set_xlabel("Time $(s)$", fontsize =14)
    ax14.set_title(r"dy $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax14.grid()
    plt.show()
    
    # z-Velocity plot
    fig15, ax15 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax15.plot(masterDict[key]['tt'], masterDict[key]['dz'], label = key)
    ax15.set_xlabel("Time $(s)$", fontsize =14)
    ax15.set_title(r"dz $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax15.grid()
    plt.show()
    
    ######################### Quaternion Plots #########################
    # sq plot
    fig16, ax16 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax16.plot(masterDict[key]['tt'], masterDict[key]['sq'], label = key)
    ax16.set_xlabel("Time $(s)$", fontsize =14)
    ax16.set_title(r"sq $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax16.grid()
    plt.show()
    
    # v1 plot
    fig17, ax17 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax17.plot(masterDict[key]['tt'], masterDict[key]['v1'], label = key)
    ax17.set_xlabel("Time $(s)$", fontsize =14)
    ax17.set_title(r"v1 $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax17.grid()
    plt.show()
    
    # v2 plot
    fig18, ax18 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax18.plot(masterDict[key]['tt'], masterDict[key]['v2'], label = key)
    ax18.set_xlabel("Time $(s)$", fontsize =14)
    ax18.set_title(r"v2 $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax18.grid()
    plt.show()
    
    # v3 plot
    fig19, ax19 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax19.plot(masterDict[key]['tt'], masterDict[key]['v3'], label = key)
    ax19.set_xlabel("Time $(s)$", fontsize =14)
    ax19.set_title(r"v3 $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax19.grid()
    plt.show()
    
    ######################### Angular Velocity Plots #########################
    # dw1 plot
    fig20, ax20 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax20.plot(masterDict[key]['tt'], masterDict[key]['dw1'], label = key)
    ax20.set_xlabel("Time $(s)$", fontsize =14)
    ax20.set_title(r"dw1 $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax20.grid()
    plt.show()
    
    # dw2 plot
    fig21, ax21 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax21.plot(masterDict[key]['tt'], masterDict[key]['dw2'], label = key)
    ax21.set_xlabel("Time $(s)$", fontsize =14)
    ax21.set_title(r"dw2 $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax21.grid()
    plt.show()
    
    # dw3 plot
    fig22, ax22 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax22.plot(masterDict[key]['tt'], masterDict[key]['dw3'], label = key)
    ax22.set_xlabel("Time $(s)$", fontsize =14)
    ax22.set_title(r"dw3 $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax22.grid()
    plt.show()
    
    ######################### Other Plots #########################
    # Total Cost Plot
    fig2, ax2 = plt.subplots()
    plt.style.use('default')
    for key in masterDict:
        ax2.plot(masterDict[key]['tt'], masterDict[key]['cost'], label = key)
    ax2.set_ylabel(r'$(x(k) - x_d)^T Q (x(k) - x_d) + u(k)^T R u(k)$', fontsize =14)
    ax2.set_xlabel("Time $(s)$", fontsize =14)
    ax2.set_title(r"Cost $(j_{\max} = %d)$" % maxIter, fontsize =14)
    #ax2.legend(loc = 'upper right', fontsize =12, title="Maximum Iterations")
    ax2.grid()
    #plt.savefig('PythonPlots/plot1.eps', format = 'eps', bbox_inches='tight')
    plt.show()
    
    # Average Cost Plot
    fig3, ax3 = plt.subplots()
    plt.style.use('default')
    ax3.plot(avgTimescale, avgCost, label = key)
    ax3.set_ylabel(r'$(x(k) - x_d)^T Q (x(k) - x_d) + u(k)^T R u(k)$', fontsize =14)
    ax3.set_xlabel("Time $(s)$", fontsize =14)
    ax3.set_title(r"Average Cost $(j_{\max} = %d)$" % maxIter, fontsize =14)
    #ax2.legend(loc = 'upper right', fontsize =12, title="Maximum Iterations")
    ax3.grid()
    #plt.savefig('PythonPlots/plot1.eps', format = 'eps', bbox_inches='tight')
    plt.show()
    
    # Average Error Plot
    fig4, ax4 = plt.subplots()
    plt.style.use('default')
    ax4.set_yscale('log')
    ax4.plot(avgTimescale, avgErrorSt, label = r'$ \| x(k) - x_d \|_2$')
    ax4.plot(avgTimescale, avgErrorCon, label = r'$ \| u(k) \|_2$')
    # ax4.set_ylabel(r'$ \| x(k) - x_d \|_2$', fontsize =14)
    ax4.set_xlabel("Time $(s)$", fontsize =14)
    ax4.set_title(r"Average Error $(j_{\max} = %d)$" % maxIter, fontsize =14)
    ax4.legend(fontsize =12)
    ax4.grid()
    #plt.savefig('PythonPlots/plot1.eps', format = 'eps', bbox_inches='tight')
    plt.show()
    
# %%
fig30, ax30 = plt.subplots()
plt.style.use('default')
for key in itersDict:
    ax30.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgCost'], label = key)
# ax30.set_yscale('log')
ax30.set_ylabel(r'$(x(k) - x_d)^T Q (x(k) - x_d) + u(k)^T R u(k)$', fontsize =14)
ax30.set_xlabel("Time $(s)$", fontsize =14)
ax30.set_title(r"Average Cost", fontsize =14)
ax30.legend(loc = 'upper right', fontsize =12, title="Maximum Iterations")
ax30.grid()
#plt.savefig('PythonPlots/plot1.eps', format = 'eps', bbox_inches='tight')
plt.show()
    
fig31, ax31 = plt.subplots()
plt.style.use('default')
for key in itersDict:
    ax31.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgError'], label = key)
# ax31.set_yscale('log')
ax31.set_ylabel(r'$ \| (x(k),u(k)) - (x_d,u_d) \|_2$', fontsize =14)
ax31.set_xlabel("Time $(s)$", fontsize =14)
ax31.set_title(r"Average Error", fontsize =14)
ax31.legend(loc = 'upper right', fontsize =12, title="Maximum Iterations")
ax31.grid()
#plt.savefig('PythonPlots/plot1.eps', format = 'eps', bbox_inches='tight')
plt.show()

fig32, ax32 = plt.subplots()
plt.style.use('default')
for key in itersDict:
    ax32.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgCost'], label = key)
ax32.set_yscale('log')
ax32.set_ylabel(r'$(x(k) - x_d)^T Q (x(k) - x_d) + u(k)^T R u(k)$', fontsize =14)
ax32.set_xlabel("Time $(s)$", fontsize =14)
ax32.set_title(r"Average Cost", fontsize =14)
ax32.legend(loc = 'upper right', fontsize =12, title="Maximum Iterations")
ax32.grid()
#plt.savefig('PythonPlots/plot1.eps', format = 'eps', bbox_inches='tight')
plt.show()
    
fig33, ax33 = plt.subplots()
plt.style.use('default')
for key in itersDict:
    ax33.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgError'], label = key)
ax33.set_yscale('log')
ax33.set_ylabel(r'$ \| (x(k),u(k)) - (x_d,u_d) \|_2$', fontsize =14)
ax33.set_xlabel("Time $(s)$", fontsize =14)
ax33.set_title(r"Average Error", fontsize =14)
ax33.legend(loc = 'upper right', fontsize =12, title="Maximum Iterations")
ax33.grid()
#plt.savefig('PythonPlots/plot1.eps', format = 'eps', bbox_inches='tight')
plt.show()
    