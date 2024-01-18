"""
Author: Gabriel Behrendt
Date: Tue Jan  9 11:21:36 2024
"""

import csv
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os
from numpy import linalg as LA

readFile = "C:/Users/15633/OneDrive - Georgia Institute of Technology/UFstuff/myPapers/Paper4/Results/maxIter3.csv"
readFolder ="C:/Users/15633/OneDrive - Georgia Institute of Technology/UFstuff/myPapers/Paper4/Results/"

masterDict = {}

directory = os.fsencode(readFolder)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"): 
        readFile = directory.decode("utf-8") + filename
        print(readFile)
        
        # Open the CSV file for reading
        with open(readFile, mode='r') as file2:
            # Create a CSV reader with DictReader
            csv_reader = csv.DictReader(file2)
         
            # Initialize an empty list to store the dictionaries
            data_list = []
         
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
            thrust1 = [0]
            thrust2 = [0]
            thrust3 =[0]
            tau1 = [0]
            tau2 =[0]
            tau3 = [0]
    
    
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
                
                if numRows < numStates:
                    x0.append(float(row["x0"]))
                    
                # read in state trajectories
                x.append(float(row["x"]))
                y.append(float(row["y"]))
                z.append(float(row["z"]))
                dx.append(float(row["xdot"]))
                dy.append(float(row["ydot"]))
                dz.append(float(row["zdot"]))
                sq.append(float(row["sq"]))
                v1.append(float(row["v1"]))
                v2.append(float(row["v2"]))
                v3.append(float(row["v3"]))
                dw1.append(float(row["dw1"]))
                dw2.append(float(row["dw2"]))
                dw3.append(float(row["dw3"]))
                
                # read in control
                thrust1.append(float(row["thrust1"]))
                thrust2.append(float(row["thrust2"]))
                thrust3.append(float(row["thrust3"]))
                tau1.append(float(row["tau1"]))
                tau2.append(float(row["tau2"]))
                tau3.append(float(row["tau3"]))
                
                
                numRows = numRows+1
        # Append initial state to lists
        x.insert(0,x0[0])
        y.insert(0,x0[1])
        z.insert(0,x0[2])
        dx.insert(0,x0[3])
        dy.insert(0,x0[4])
        dz.insert(0,x0[5])
        sq.insert(0,x0[6])
        v1.insert(0,x0[7])
        v2.insert(0,x0[8])
        v3.insert(0,x0[9])
        dw1.insert(0,x0[10])
        dw2.insert(0,x0[11])
        dw3.insert(0,x0[12])
    
        # store time axis values
        tt = np.linspace(0,numLoops*ts,numLoops+1)
        
        # Calculate Objective Function Value
        Q = np.diag([posCost, posCost, posCost, velCost, velCost, velCost,
                     quatCost, quatCost, quatCost, quatCost, angularCost, angularCost, angularCost])
        R = np.diag([thrustCost, thrustCost, thrustCost, torqueCost, torqueCost, torqueCost])
        
        # Compute error
        xd = np.array([0,0,0,0,0,0,1,0,0,0,0,0,0])
        errorSt = []
        cost = []
        for i in range(numLoops+1):
            curSt = np.array([x[i], y[i], z[i], dx[i], dy[i], dz[i], sq[i], v1[i], v2[i], v3[i], dw1[i], dw2[i], dw3[i]])
            curCon = np.array([ thrust1[i], thrust2[i], thrust3[i], tau1[i], tau2[i], tau3[i] ])
            errorSt.append( LA.norm(curSt - xd) )
            
            cost.append( curSt.T @ Q @ curSt + curCon.T @ R @ curCon)
        
        totCost = sum(cost)
        print(totCost)
        
        # change maxIter 1000 to the optimal solution
        if maxIter == 1000:
            maxIter = 'Optimal'
        
        # Store data in a dictionary
        testDict = {'x':x,'y':y,'z':z,
                    'dx':dx,'dy':dy,'dz':dz,
                    'sq':sq,'v1':v1,'v2':v2,'v3':v3,
                    'dw1':dw1,'dw2':dw2,'dw3':dw3,
                    'thrust1':thrust1,'thrust2':thrust2,'thrust3':thrust3,
                    'tau1':tau1,'tau2':tau2,'tau3':tau3,
                    'maxIter':maxIter,'ts':ts, 'N':N, 'numLoops':numLoops,
                    'posCost':posCost, 'velCost':velCost, 'quatCost':quatCost, 'angularCost':angularCost,
                    'thrustCost':thrustCost, 'torqueCost':torqueCost, 'thrustMax':thrustMax, 'torqueMax':torqueMax,
                    'tt':tt, 'error':errorSt, 'cost':cost, 'totCost':totCost
                    }
        masterDict[maxIter] = testDict
        continue
    else:
        continue


# Error
fig1, ax1 = plt.subplots()
plt.style.use('default')
for key in masterDict:
    ax1.plot(masterDict[key]['tt'], masterDict[key]['error'], label = key)
ax1.set_ylabel(r'$ \|\| x(k) - x_d \|\|_2$', fontsize =14)
ax1.set_xlabel("Time $(s)$", fontsize =14)
ax1.set_title(f"Error", fontsize =14)
ax1.legend(loc = 'upper right', fontsize =12, title="Maximum Iterations")
ax1.grid()
#plt.savefig('PythonPlots/plot1.eps', format = 'eps', bbox_inches='tight')
plt.show()

# Error
fig2, ax2 = plt.subplots()
plt.style.use('default')
for key in masterDict:
    ax2.plot(masterDict[key]['tt'], masterDict[key]['cost'], label = key)
ax2.set_ylabel(r'$x(k)^T Q x(k) + u(k)^T R u(k)$', fontsize =14)
ax2.set_xlabel("Time $(s)$", fontsize =14)
ax2.set_title(f"Cost", fontsize =14)
ax2.legend(loc = 'upper right', fontsize =12, title="Maximum Iterations")
ax2.grid()
#plt.savefig('PythonPlots/plot1.eps', format = 'eps', bbox_inches='tight')
plt.show()

# Total Cost
fig2, ax2 = plt.subplots()
plt.style.use('default')
for key in masterDict:
    print(key)
    # print(masterDict[key]['totCost'])
    ax2.bar(str(masterDict[key]['maxIter']), masterDict[key]['totCost'], label = key)
ax2.set_ylabel(r'$\sum^{N}_{k=1}  x(k)^T Q x(k) + u(k)^T R u(k)$', fontsize =14)
ax2.set_xlabel("Maiximum Iterations", fontsize =14)
ax2.set_title(f"Total Cost", fontsize =14)
# ax1.legend(loc = 'upper right', fontsize =12, title="Maximum Iterations")
ax2.grid()
#plt.savefig('PythonPlots/plot1.eps', format = 'eps', bbox_inches='tight')
plt.show()








