import csv
import scipy.io
from matplotlib import ticker
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

whichFolder = 2

savePlots = False
constraintType = "Euler"
samplingTime = "10"

iters = ["5","6","7","8","9","10","15","50","100","125","150","175","200","300","400","500","1000","10000"] # zResults

itersDict = {}
for maxIter in iters:
    print("%%%%%%%%%%",maxIter,"%%%%%%%%%%")
    if whichFolder == 0:
        readFolder = "/home/gbehrendt/CLionProjects/Satellite/finalResultsExactNoCons3/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
    elif whichFolder == 1:
        readFolder = "/home/gbehrendt/CLionProjects/Satellite/zResults/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
    elif whichFolder == 2:
        readFolder = "/home/gbehrendt/CLionProjects/Satellite/Results/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
    elif whichFolder == 3:
        readFolder = "/home/gbehrendt/CLionProjects/Satellite/newQuatResults/" + constraintType + "/ts" + samplingTime + "/maxIter" + maxIter + "/"
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
                errorTrans.append(LA.norm(curSt[0:6]))
                
                errorQuat.append(LA.norm(curSt[6:10] - xd[6:10]))
                errorAngVel.append(LA.norm(curSt[10:13]))
                errorRot.append(LA.norm(curSt[6:13] - xd[6:13]))
                
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
                        'tt':tt, 'endTime':endTime, 'errorSt':errorSt, 'errorCon':errorCon,
                        'errorPos':errorPos, 'errorVel':errorVel, 'errorTrans':errorTrans, 
                        'errorQuat':errorQuat,'errorAngVel':errorAngVel, 'errorRot':errorRot,
                        'errorThrust':errorThrust, 'errorTau':errorTau,
                        'cost':cost, 'totError':totError,'totCost':totCost, 'trial':trial,
                        'infNorm':infNorm, 'x0':x0
                        }
            if(converged == "yes"):
                masterDict[trial] = trialDict
                numConverged += 1
            else:
                masterDict[trial] = trialDict
                numNotConverged += 1
                print(trial)
                print(masterDict[trial]['infNorm'])
                print(masterDict[trial]['x0'])
            continue
        else:
            continue
        
    numTrials = numConverged + numNotConverged
    print("# Converged:", numConverged)
    print("# Not Converged:", numNotConverged)
    print("Total Run:", numTrials)
    
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
    
    
    
    for key in masterDict:
        allErrorSt.append(masterDict[key]['errorSt'])
        allErrorCon.append(masterDict[key]['errorCon'])
        allError.append(masterDict[key]['totError'])
        allCost.append(masterDict[key]['cost'])
        
        allErrorPos.append(masterDict[key]['errorPos'])
        allErrorVel.append(masterDict[key]['errorVel'])
        allErrorTrans.append(masterDict[key]['errorTrans'])
        
        allErrorQuat.append(masterDict[key]['errorQuat'])
        allErrorAngVel.append(masterDict[key]['errorAngVel'])
        allErrorRot.append(masterDict[key]['errorRot'])
        
        allErrorThrust.append(masterDict[key]['errorThrust'])
        allErrorTau.append(masterDict[key]['errorTau'])
    
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
    
    
    itersDict[maxIter] = {'avgErrorSt':avgErrorSt, 'avgErrorCon':avgErrorCon, 'avgError':avgError, 'avgCost':avgCost,
                          'avgErrorPos':avgErrorPos, 'avgErrorVel':avgErrorVel, 'avgErrorTrans':avgErrorTrans,
                          'avgErrorQuat':avgErrorQuat, 'avgErrorAngVel':avgErrorAngVel, 'avgErrorRot':avgErrorRot,
                          'avgErrorThrust':avgErrorThrust, 'avgErrorTau':avgErrorTau,
                          'totAvgCost':totAvgCost,'avgTimescale':avgTimescale}
    
    
    
    # # %%
    # ######################### Position Plots #################################
    # # x-Position plot
    # fig10, ax10 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax10.plot(masterDict[key]['tt'], masterDict[key]['x'], label = key)
    # ax10.set_xlabel("Time $(s)$", fontsize =14)
    # ax10.set_ylabel("$\delta x^{\mathcal{O}} \ (km)$", fontsize =14)
    # # ax10.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax10.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/xPos.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    
    # # y-Position plot
    # fig11, ax11 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax11.plot(masterDict[key]['tt'], masterDict[key]['y'], label = key)
    # ax11.set_xlabel("Time $(s)$", fontsize =14)
    # ax11.set_ylabel("$\delta y^{\mathcal{O}} \ (km)$", fontsize =14)
    # # ax11.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax11.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/yPos.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    
    # # z-Position plot
    # fig12, ax12 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax12.plot(masterDict[key]['tt'], masterDict[key]['z'], label = key)
    # ax12.set_xlabel("Time $(s)$", fontsize =14)
    # ax12.set_ylabel("$\delta z^{\mathcal{O}} \ (km)$", fontsize =14)
    # # ax12.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax12.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/zPos.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    

    # # %%
    # ######################### Velocity Plots #########################
    # # x-Velocity plot
    # fig13, ax13 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax13.plot(masterDict[key]['tt'], masterDict[key]['dx'], label = key)
    # ax13.set_xlabel("Time $(s)$", fontsize =14)
    # ax13.set_ylabel(r"$\delta \dot{x}^{\mathcal{O}} \ (km/s)$", fontsize =14)
    # # ax13.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax13.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/xVel.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    
    # # y-Velocity plot
    # fig14, ax14 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax14.plot(masterDict[key]['tt'], masterDict[key]['dy'], label = key)
    # ax14.set_xlabel("Time $(s)$", fontsize =14)
    # ax14.set_ylabel(r"$\delta \dot{y}^{\mathcal{O}} \ (km/s)$", fontsize =14)
    # # ax14.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax14.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/yVel.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    
    # # z-Velocity plot
    # fig15, ax15 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax15.plot(masterDict[key]['tt'], masterDict[key]['dz'], label = key)
    # ax15.set_xlabel("Time $(s)$", fontsize =14)
    # ax15.set_ylabel(r"$\delta \dot{z}^{\mathcal{O}} \ (km/s)$", fontsize =14)
    # # ax15.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax15.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/zVel.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    # # %%
    # ######################## Quaternion Plots #########################
    # # sq plot
    # fig16, ax16 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax16.plot(masterDict[key]['tt'], masterDict[key]['sq'], label = key)
    # ax16.set_xlabel("Time $(s)$", fontsize =14)
    # ax16.set_ylabel(r"$\delta \eta^{\mathcal{O}}_{\mathcal{D}}$", fontsize =14)
    # # ax16.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax16.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/sq.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    
    # # v1 plot
    # fig17, ax17 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax17.plot(masterDict[key]['tt'], masterDict[key]['v1'], label = key)
    # ax17.set_xlabel("Time $(s)$", fontsize =14)
    # ax17.set_ylabel(r"$\delta \rho^{\mathcal{O}}_{\mathcal{D},1}$", fontsize =14)
    # # ax17.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax17.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/v1.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    
    # # v2 plot
    # fig18, ax18 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax18.plot(masterDict[key]['tt'], masterDict[key]['v2'], label = key)
    # ax18.set_xlabel("Time $(s)$", fontsize =14)
    # ax18.set_ylabel(r"$\delta \rho^{\mathcal{O}}_{\mathcal{D},2}$", fontsize =14)
    # # ax18.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax18.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/v2.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    
    # # v3 plot
    # fig19, ax19 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax19.plot(masterDict[key]['tt'], masterDict[key]['v3'], label = key)
    # ax19.set_xlabel("Time $(s)$", fontsize =14)
    # ax19.set_ylabel(r"$\delta \rho^{\mathcal{O}}_{\mathcal{D},3}$", fontsize =14)
    # # ax19.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax19.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/v3.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    # # %%
    # ######################### Angular Velocity Plots #########################
    # # dw1 plot
    # fig20, ax20 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax20.plot(masterDict[key]['tt'], masterDict[key]['dw1'], label = key)
    # ax20.set_xlabel("Time $(s)$", fontsize =14)
    # ax20.set_ylabel(r"$\delta \omega^{\mathcal{O}}_{\mathcal{OD},x} \ (rad/s)$", fontsize =14)
    # # ax20.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax20.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/dw1.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    
    # # dw2 plot
    # fig21, ax21 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax21.plot(masterDict[key]['tt'], masterDict[key]['dw2'], label = key)
    # ax21.set_xlabel("Time $(s)$", fontsize =14)
    # ax21.set_ylabel(r"$\delta \omega^{\mathcal{O}}_{\mathcal{OD},y} \ (rad/s)$", fontsize =14)
    # # ax21.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax21.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/dw2.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    
    # # dw3 plot
    # fig22, ax22 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     ax22.plot(masterDict[key]['tt'], masterDict[key]['dw3'], label = key)
    # ax22.set_xlabel("Time $(s)$", fontsize =14)
    # ax22.set_ylabel(r"$\delta \omega^{\mathcal{O}}_{\mathcal{OD},z} \ (rad/s)$", fontsize =14)
    # # ax22.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax22.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/dw3.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    # # %%
    # ######################### Thrust Plots #########################
    # ts = 10
    
    # fig23, ax23 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     plotThrust1 = []
    #     for i in masterDict[key]['thrust1']:
    #         for ele in range(ts):
    #             plotThrust1.append(i)
    #         masterDict[key]['ttPlot'] = range(0,int(masterDict[key]['tt'][-1])+ts)
    #     masterDict[key]['plotThrust1'] = plotThrust1
    # for key in masterDict:
    #     # ax23.plot(masterDict[key]['tt'], masterDict[key]['thrust1'], label = key)
    #     ax23.plot(masterDict[key]['ttPlot'], masterDict[key]['plotThrust1'], label = key)
    # ax23.set_xlabel("Time $(s)$", fontsize =14)
    # ax23.set_ylabel(r"$F^{\mathcal{D}}_{x} \ (N)$", fontsize =14)
    # # ax23.set_title(r"$j_{\max} = %d$" % maxIter, fontsize =14)
    # ax23.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/thrust1.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    
    # fig24, ax24 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     plotThrust2 = []
    #     for i in masterDict[key]['thrust2']:
    #         for ele in range(ts):
    #             plotThrust2.append(i)
    #     masterDict[key]['plotThrust2'] = plotThrust2
    # for key in masterDict:
    #     ax24.plot(masterDict[key]['ttPlot'], masterDict[key]['plotThrust2'], label = key)
    # ax24.set_xlabel("Time $(s)$", fontsize =14)
    # ax24.set_ylabel(r"$F^{\mathcal{D}}_{y} \ (N)$", fontsize =14)
    # # ax24.set_title(r"Thrust 2 $(j_{\max} = %d) \ (N)$" % maxIter, fontsize =14)
    # ax24.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/thrust2.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    
    # fig25, ax25 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     plotThrust3 = []
    #     for i in masterDict[key]['thrust3']:
    #         for ele in range(ts):
    #             plotThrust3.append(i)
    #     masterDict[key]['plotThrust3'] = plotThrust3
    # for key in masterDict:
    #     ax25.plot(masterDict[key]['ttPlot'], masterDict[key]['plotThrust3'], label = key)
    # ax25.set_xlabel("Time $(s)$", fontsize =14)
    # ax25.set_ylabel(r"$F^{\mathcal{D}}_{z} \ (N)$", fontsize =14)
    # # ax25.set_title(r"Thrust 3 $(j_{\max} = %d)$" % maxIter, fontsize =14)
    # ax25.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/thrust3.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    # # %%
    # ######################### Torque Plots #########################
    # fig26, ax26 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     plotTau1 = []
    #     for i in masterDict[key]['tau1']:
    #         for ele in range(ts):
    #             plotTau1.append(i)
    #     masterDict[key]['plotTau1'] = plotTau1
    # for key in masterDict:
    #     ax26.plot(masterDict[key]['ttPlot'], masterDict[key]['plotTau1'], label = key)
    # # for key in masterDict:
    # #     ax26.plot(masterDict[key]['tt'], masterDict[key]['tau1'], label = key)
    # ax26.set_xlabel("Time $(s)$", fontsize =14)
    # ax26.set_ylabel(r"$\tau^{\mathcal{D}}_{x} \ (rad/s^2)$", fontsize =14)
    # # ax26.set_title(r"Torque 1 $(j_{\max} = %d)$" % maxIter, fontsize =14)
    # ax26.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/tau1.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    
    # fig27, ax27 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     plotTau2 = []
    #     for i in masterDict[key]['tau2']:
    #         for ele in range(ts):
    #             plotTau2.append(i)
    #     masterDict[key]['plotTau2'] = plotTau2
    # for key in masterDict:
    #     ax27.plot(masterDict[key]['ttPlot'], masterDict[key]['plotTau2'], label = key)
    # # for key in masterDict:
    # #     ax27.plot(masterDict[key]['tt'], masterDict[key]['tau2'], label = key)
    # ax27.set_xlabel("Time $(s)$", fontsize =14)
    # ax27.set_ylabel(r"$\tau^{\mathcal{D}}_{y} \ (rad/s^2)$", fontsize =14)
    # # ax27.set_title(r"Torque 2 $(j_{\max} = %d)$" % maxIter, fontsize =14)
    # ax27.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/tau2.eps', format = 'eps', bbox_inches='tight')
    # plt.show()
    
    # fig28, ax28 = plt.subplots()
    # plt.style.use('default')
    # for key in masterDict:
    #     plotTau3 = []
    #     for i in masterDict[key]['tau3']:
    #         for ele in range(ts):
    #             plotTau3.append(i)
    #     masterDict[key]['plotTau3'] = plotTau3
    # for key in masterDict:
    #     ax28.plot(masterDict[key]['ttPlot'], masterDict[key]['plotTau3'], label = key)
    # # for key in masterDict:
    # #     ax28.plot(masterDict[key]['tt'], masterDict[key]['tau3'], label = key)
    # ax28.set_xlabel("Time $(s)$", fontsize =14)
    # ax28.set_ylabel(r"$\tau^{\mathcal{D}}_{z} \ (rad/s^2)$", fontsize =14)
    # # ax28.set_title(r"Torque 3 $(j_{\max} = %d)$" % maxIter, fontsize =14)
    # ax28.grid()
    # if savePlots == True:
    #     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/maxIter'+str(maxIter)+'/tau3.eps', format = 'eps', bbox_inches='tight')
    # plt.show()

    
# %%

# numBars = len(itersDict)
# colormap = plt.cm.get_cmap('jet', numBars)
# myColor =[colormap(i) for i in range(numBars)]
# myColor = [ "1FCC6D","4B8BEB", "D64215", "F58201", "0C7500", "820099", "00D6DB", "A87CD6", "4502EB", "EB1189", "9E8001" ]
myColor = ["01CC58", "0075EB", "D64215", "F58201", "1D7500", "820099", "01B5DB", "FF02EB","00D3B4", "F5001E",
           "F0C055", "9E8001", "A87CD6", "BB4E8F", "B0C700", "DB8E73", "64FAC4", "001CBD","EB1189"]

for i in range(len(myColor)):
    h = myColor[i]
    myColor[i] = tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))
    # print(myColor[i])
    # print('RGB =', tuple(int(h[i:i+2], 16) for i in (0, 2, 4)))


# %%

# fig30, ax30 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     ax30.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgCost'], label = key)
# # ax30.set_yscale('log')
# ax30.set_ylabel(r'$(x(k) - x_d)^T Q (x(k) - x_d) + u(k)^T R u(k)$', fontsize =14)
# ax30.set_xlabel("Time $(s)$", fontsize =14)
# ax30.set_title(r"Average Cost", fontsize =14)
# ax30.legend(loc = 'upper right', fontsize =12, title="Maximum Iterations")
# ax30.grid()
# plt.show()
    
# fig31, ax31 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     ax31.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgError'], label = key)
# # ax31.set_yscale('log')
# ax31.set_ylabel(r'$ \| (x(k),u(k)) - (x_d,u_d) \|_2$', fontsize =14)
# ax31.set_xlabel("Time $(s)$", fontsize =14)
# ax31.set_title(r"Average Error", fontsize =14)
# ax31.legend(loc = 'upper right', fontsize =12, title="Maximum Iterations")
# ax31.grid()
# plt.show()

# i = 0
# fig32, ax32 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     lbl = key
#     if lbl == 10000 or lbl == '10000':
#         lbl = 'Optimal'
#     ax32.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgCost'], color = myColor[i], label = lbl)
#     i+=1
# ax32.set_yscale('log')
# ax32.set_ylabel(r'$(x(k) - x_d)^T Q (x(k) - x_d) + u(k)^T R u(k)$', fontsize =14)
# ax32.set_xlabel("Time $(s)$", fontsize =14)
# # ax32.set_title(r"Average Cost", fontsize =14)
# leg = ax32.legend(fontsize =10,loc = "upper right", ncol = 3)
# leg.set_title(r'$j_{\max}$',prop={'size':13})
# ax32.grid()
# if savePlots == True:
#     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgCost.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# i = 0
# fig33, ax33 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     lbl = key
#     if lbl == 10000 or lbl == '10000':
#         lbl = 'Optimal'
#     ax33.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgError'], color = myColor[i], label = lbl)
#     i+=1
# ax33.set_yscale('log')
# ax33.set_ylabel(r'$ \| (x(k),u(k)) - (x_d,u_d) \|_2$', fontsize =14)
# ax33.set_xlabel("Time $(s)$", fontsize =14)
# # ax33.set_title(r"Average Error", fontsize =14)
# leg = ax33.legend(fontsize =10,loc = "upper right", ncol = 3)
# leg.set_title(r'$j_{\max}$',prop={'size':12})
# ax33.grid()
# if savePlots == True:
#     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgError.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# i = 0
# fig35, ax35 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     lbl = key
#     if lbl == 10000 or lbl == '10000':
#         lbl = 'Optimal'
#     ax35.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorSt'], color = myColor[i], label = lbl)
#     i+=1
# ax35.set_yscale('log')
# ax35.set_ylabel(r'$ \| x(k)- x_d \|_2$', fontsize =14)
# ax35.set_xlabel("Time $(s)$", fontsize =14)
# # ax35.set_title(r"Average State Error", fontsize =14)
# leg = ax35.legend(fontsize =10,loc = "upper right", ncol = 3)
# leg.set_title(r'$j_{\max}$',prop={'size':13})
# ax35.grid()
# if savePlots == True:
#     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgErrorState.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# i = 0
# fig34, ax34 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     lbl = key
#     if lbl == 10000 or lbl == '10000':
#         lbl = 'Optimal'
#     ax34.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorCon'], color = myColor[i], label = lbl)
#     i+=1
# ax34.set_yscale('log')
# ax34.set_ylabel(r'$ \| u(k) \|_2$', fontsize =14)
# ax34.set_xlabel("Time $(s)$", fontsize =14)
# # ax34.set_title(r"Average Control", fontsize =14)
# # ax34.legend(loc = 'lower left', fontsize =9, title=r'$j_{\max}$', ncols = 4)
# leg = ax34.legend(fontsize =9,loc = "upper right", ncol = 4)
# leg.set_title(r'$j_{\max}$',prop={'size':12})
# ax34.grid()
# if savePlots == True:
#     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgErrorControl.eps', format = 'eps', bbox_inches='tight')
# plt.show()




# # %%
# i = 0
# fig40, ax40 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     lbl = key
#     if lbl == 10000 or lbl == '10000':
#         lbl = 'Optimal'
#     ax40.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorPos'], color = myColor[i], label = lbl)
#     i+=1
# ax40.set_yscale('log')
# ax40.set_ylabel(r'$ \| \delta r^{\mathcal{O}} \|_2$', fontsize =14)
# ax40.set_xlabel("Time $(s)$", fontsize =14)
# leg = ax40.legend(fontsize =10,loc = "upper right", ncol = 3)
# leg.set_title(r'$j_{\max}$',prop={'size':12})
# ax40.grid()
# if savePlots == True:
#     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgPosError.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# i = 0
# fig41, ax41 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     lbl = key
#     if lbl == 10000 or lbl == '10000':
#         lbl = 'Optimal'
#     ax41.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorVel'], color = myColor[i], label = lbl)
#     i+=1
# ax41.set_yscale('log')
# ax41.set_ylabel(r'$ \| \delta \dot{r}^{\mathcal{O}} \|_2$', fontsize =14)
# ax41.set_xlabel("Time $(s)$", fontsize =14)
# leg = ax41.legend(fontsize =9,loc = "upper right", ncol = 3)
# leg.set_title(r'$j_{\max}$',prop={'size':12})
# ax41.grid()
# if savePlots == True:
#     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgVelError.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# i = 0
# fig42, ax42 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     lbl = key
#     if lbl == 10000 or lbl == '10000':
#         lbl = 'Optimal'
#     ax42.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorQuat'], color = myColor[i], label = lbl)
#     i+=1
# ax42.set_yscale('log')
# ax42.set_ylabel(r'$ \| \delta q_{\mathcal{D}}^{\mathcal{O}} - q^I \|_2$', fontsize =14)
# ax42.set_xlabel("Time $(s)$", fontsize =14)
# leg = ax42.legend(fontsize =10,loc = "upper right", ncol = 3)
# leg.set_title(r'$j_{\max}$',prop={'size':12})
# ax42.grid()
# if savePlots == True:
#     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgQuatError.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# i = 0
# fig43, ax43 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     lbl = key
#     if lbl == 10000 or lbl == '10000':
#         lbl = 'Optimal'
#     ax43.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorAngVel'], color = myColor[i], label = lbl)
#     i+=1
# ax43.set_yscale('log')
# ax43.set_ylabel(r'$ \| \delta \omega_{\mathcal{D}}^{\mathcal{O}} \|_2$', fontsize =14)
# ax43.set_xlabel("Time $(s)$", fontsize =14)
# leg = ax43.legend(fontsize =10,loc = "upper right", ncol = 3)
# leg.set_title(r'$j_{\max}$',prop={'size':12})
# ax43.grid()
# if savePlots == True:
#     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgAngVelError.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# i = 0
# fig44, ax44 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     lbl = key
#     if lbl == 10000 or lbl == '10000':
#         lbl = 'Optimal'
#     ax44.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorTrans'], color = myColor[i], label = lbl)
#     i+=1
# # ax44.set_yscale('log')
# ax44.set_ylabel(r'$ \| (\delta r^{\mathcal{O}}, \delta \dot{r}^{\mathcal{O}}) \|_2$', fontsize =14)
# ax44.set_xlabel("Time $(s)$", fontsize =14)
# leg = ax44.legend(fontsize =10,loc = "upper right", ncol = 3)
# leg.set_title(r'$j_{\max}$',prop={'size':12})
# ax44.grid()
# if savePlots == True:
#     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgTransError.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# i = 0
# fig45, ax45 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     lbl = key
#     if lbl == 10000 or lbl == '10000':
#         lbl = 'Optimal'
#     ax45.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorRot'], color = myColor[i], label = lbl)
#     i+=1
# # ax44.set_yscale('log')
# ax45.set_ylabel(r'$ \| (\delta q_{\mathcal{D}}^{\mathcal{O}} - q^I, \delta \omega_{\mathcal{D}}^{\mathcal{O}}) \|_2$', fontsize =14)
# ax45.set_xlabel("Time $(s)$", fontsize =14)
# leg = ax45.legend(fontsize =10,loc = "upper right", ncol = 3)
# leg.set_title(r'$j_{\max}$',prop={'size':12})
# ax45.grid()
# if savePlots == True:
#     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgRotError.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# i = 0
# fig46, ax46 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     lbl = key
#     if lbl == 10000 or lbl == '10000':
#         lbl = 'Optimal'
#     ax46.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorThrust'], color = myColor[i], label = lbl)
#     i+=1
# # ax46.set_yscale('log')
# ax46.set_ylabel(r'$ \| F^{\mathcal{D}}_d \|_2$', fontsize =14)
# ax46.set_xlabel("Time $(s)$", fontsize =14)
# plt.ticklabel_format(axis='y', style='sci', scilimits=(2,4))
# leg = ax46.legend(fontsize =10,loc = "upper right", ncol = 3)
# leg.set_title(r'$j_{\max}$',prop={'size':12})
# ax46.grid()
# if savePlots == True:
#     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgErrorThrust.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# i = 0
# fig47, ax47 = plt.subplots()
# plt.style.use('default')
# for key in itersDict:
#     lbl = key
#     if lbl == 10000 or lbl == '10000':
#         lbl = 'Optimal'
#     ax47.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorTau'], color = myColor[i], label = lbl)
#     i+=1
# # ax47.set_yscale('log')
# ax47.set_ylabel(r'$ \| \tau^{D} \|_2$', fontsize =14)
# ax47.set_xlabel("Time $(s)$", fontsize =14)
# plt.ticklabel_format(axis='y', style='sci', scilimits=(2,4))
# leg = ax47.legend(fontsize =10,loc = "upper right", ncol = 3)
# leg.set_title(r'$j_{\max}$',prop={'size':12})
# ax47.grid()
# if savePlots == True:
#     plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgErrorTau.eps', format = 'eps', bbox_inches='tight')
# plt.show()

# %%

if whichFolder == 2:
    i = 0
    fig32, ax32 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax32.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgCost'], color = myColor[i], label = lbl)
        i+=1
    ax32.set_yscale('log')
    ax32.set_ylabel(r'$(x(k) - x_d)^T Q (x(k) - x_d) + u(k)^T R u(k)$', fontsize =14)
    ax32.set_xlabel("Time $(s)$", fontsize =14)
    # ax32.set_title(r"Average Cost", fontsize =14)
    leg = ax32.legend(fontsize =10,loc = "lower left", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':13})
    ax32.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgCost.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig33, ax33 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax33.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgError'], color = myColor[i], label = lbl)
        i+=1
    ax33.set_yscale('log')
    ax33.set_ylabel(r'$ \| (x(k),u(k)) - (x_d,u_d) \|_2$', fontsize =14)
    ax33.set_xlabel("Time $(s)$", fontsize =14)
    ax33.set_ylim([0,15])
    # ax33.set_title(r"Average Error", fontsize =14)
    leg = ax33.legend(fontsize = 10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax33.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig35, ax35 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax35.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorSt'], color = myColor[i], label = lbl)
        i+=1
    ax35.set_yscale('log')
    ax35.set_ylabel(r'$ \| x(k)- x_d \|_2$', fontsize =14)
    ax35.set_xlabel("Time $(s)$", fontsize =14)
    ax35.set_ylim([0,50])
    # ax35.set_title(r"Average State Error", fontsize =14)
    leg = ax35.legend(fontsize = 10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':13})
    ax35.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgErrorState.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig34, ax34 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax34.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorCon'], color = myColor[i], label = lbl)
        i+=1
    ax34.set_yscale('log')
    ax34.set_ylabel(r'$ \| u(k) \|_2$', fontsize =14)
    ax34.set_xlabel("Time $(s)$", fontsize =14)
    ax34.set_ylim([0,10])
    # ax34.set_title(r"Average Control", fontsize =14)
    # ax34.legend(loc = 'lower left', fontsize =9, title=r'$j_{\max}$', ncols = 4)
    leg = ax34.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax34.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgErrorControl.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig40, ax40 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax40.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorPos'], color = myColor[i], label = lbl)
        i+=1
    ax40.set_yscale('log')
    ax40.set_ylabel(r'$ \| \delta r^{\mathcal{O}} \|_2$', fontsize =14)
    ax40.set_xlabel("Time $(s)$", fontsize =14)
    leg = ax40.legend(fontsize =10,loc = "lower left", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax40.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgPosError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig41, ax41 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax41.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorVel'], color = myColor[i], label = lbl)
        i+=1
    ax41.set_yscale('log')
    ax41.set_ylabel(r'$ \| \delta \dot{r}^{\mathcal{O}} \|_2$', fontsize =14)
    ax41.set_xlabel("Time $(s)$", fontsize =14)
    leg = ax41.legend(fontsize =9,loc = "lower left", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax41.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgVelError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig42, ax42 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax42.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorQuat'], color = myColor[i], label = lbl)
        i+=1
    ax42.set_yscale('log')
    ax42.set_ylabel(r'$ \| \delta q_{\mathcal{D}}^{\mathcal{O}} - q^I \|_2$', fontsize =14)
    ax42.set_xlabel("Time $(s)$", fontsize =14)
    leg = ax42.legend(fontsize =10,loc = "lower left", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax42.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgQuatError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig43, ax43 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax43.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorAngVel'], color = myColor[i], label = lbl)
        i+=1
    ax43.set_yscale('log')
    ax43.set_ylabel(r'$ \| \delta \omega_{\mathcal{D}}^{\mathcal{O}} \|_2$', fontsize =14)
    ax43.set_xlabel("Time $(s)$", fontsize =14)
    leg = ax43.legend(fontsize =10,loc = "lower left", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax43.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgAngVelError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig44, ax44 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax44.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorTrans'], color = myColor[i], label = lbl)
        i+=1
    # ax44.set_yscale('log')
    ax44.set_ylabel(r'$ \| (\delta r^{\mathcal{O}}, \delta \dot{r}^{\mathcal{O}}) \|_2$', fontsize =14)
    ax44.set_xlabel("Time $(s)$", fontsize =14)
    leg = ax44.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax44.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgTransError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig45, ax45 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax45.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorRot'], color = myColor[i], label = lbl)
        i+=1
    # ax44.set_yscale('log')
    ax45.set_ylabel(r'$ \| (\delta q_{\mathcal{D}}^{\mathcal{O}} - q^I, \delta \omega_{\mathcal{D}}^{\mathcal{O}}) \|_2$', fontsize =14)
    ax45.set_xlabel("Time $(s)$", fontsize =14)
    leg = ax45.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax45.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgRotError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig46, ax46 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax46.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorThrust'], color = myColor[i], label = lbl)
        i+=1
    # ax46.set_yscale('log')
    ax46.set_ylabel(r'$ \| F^{\mathcal{D}}_d \|_2$', fontsize =14)
    ax46.set_xlabel("Time $(s)$", fontsize =14)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(2,4))
    leg = ax46.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax46.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgErrorThrust.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig47, ax47 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax47.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorTau'], color = myColor[i], label = lbl)
        i+=1
    # ax47.set_yscale('log')
    ax47.set_ylabel(r'$ \| \tau^{D} \|_2$', fontsize =14)
    ax47.set_xlabel("Time $(s)$", fontsize =14)
    ax47.set_ylim([0,3e-4])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(2,4))
    leg = ax47.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax47.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/newFigures/avgErrorTau.eps', format = 'eps', bbox_inches='tight')
    plt.show()
else:
    i = 0
    fig32, ax32 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax32.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgCost'], color = myColor[i], label = lbl)
        i+=1
    ax32.set_yscale('log')
    ax32.set_ylabel(r'$(x(k) - x_d)^T Q (x(k) - x_d) + u(k)^T R u(k)$', fontsize =14)
    ax32.set_xlabel("Time $(s)$", fontsize =14)
    # ax32.set_title(r"Average Cost", fontsize =14)
    leg = ax32.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':13})
    ax32.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgCost.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig33, ax33 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax33.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgError'], color = myColor[i], label = lbl)
        i+=1
    ax33.set_yscale('log')
    ax33.set_ylabel(r'$ \| (x(k),u(k)) - (x_d,u_d) \|_2$', fontsize =14)
    ax33.set_xlabel("Time $(s)$", fontsize =14)
    # ax33.set_title(r"Average Error", fontsize =14)
    leg = ax33.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax33.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig35, ax35 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax35.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorSt'], color = myColor[i], label = lbl)
        i+=1
    ax35.set_yscale('log')
    ax35.set_ylabel(r'$ \| x(k)- x_d \|_2$', fontsize =14)
    ax35.set_xlabel("Time $(s)$", fontsize =14)
    # ax35.set_title(r"Average State Error", fontsize =14)
    leg = ax35.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':13})
    ax35.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgErrorState.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig34, ax34 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax34.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorCon'], color = myColor[i], label = lbl)
        i+=1
    ax34.set_yscale('log')
    ax34.set_ylabel(r'$ \| u(k) \|_2$', fontsize =14)
    ax34.set_xlabel("Time $(s)$", fontsize =14)
    # ax34.set_title(r"Average Control", fontsize =14)
    # ax34.legend(loc = 'lower left', fontsize =9, title=r'$j_{\max}$', ncols = 4)
    leg = ax34.legend(fontsize =9,loc = "upper right", ncol = 4)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax34.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgErrorControl.eps', format = 'eps', bbox_inches='tight')
    plt.show()




    # %%
    i = 0
    fig40, ax40 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax40.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorPos'], color = myColor[i], label = lbl)
        i+=1
    ax40.set_yscale('log')
    ax40.set_ylabel(r'$ \| \delta r^{\mathcal{O}} \|_2$', fontsize =14)
    ax40.set_xlabel("Time $(s)$", fontsize =14)
    leg = ax40.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax40.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgPosError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig41, ax41 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax41.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorVel'], color = myColor[i], label = lbl)
        i+=1
    ax41.set_yscale('log')
    ax41.set_ylabel(r'$ \| \delta \dot{r}^{\mathcal{O}} \|_2$', fontsize =14)
    ax41.set_xlabel("Time $(s)$", fontsize =14)
    leg = ax41.legend(fontsize =9,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax41.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgVelError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig42, ax42 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax42.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorQuat'], color = myColor[i], label = lbl)
        i+=1
    ax42.set_yscale('log')
    ax42.set_ylabel(r'$ \| \delta q_{\mathcal{D}}^{\mathcal{O}} - q^I \|_2$', fontsize =14)
    ax42.set_xlabel("Time $(s)$", fontsize =14)
    leg = ax42.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax42.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgQuatError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig43, ax43 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax43.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorAngVel'], color = myColor[i], label = lbl)
        i+=1
    ax43.set_yscale('log')
    ax43.set_ylabel(r'$ \| \delta \omega_{\mathcal{D}}^{\mathcal{O}} \|_2$', fontsize =14)
    ax43.set_xlabel("Time $(s)$", fontsize =14)
    leg = ax43.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax43.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgAngVelError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig44, ax44 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax44.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorTrans'], color = myColor[i], label = lbl)
        i+=1
    # ax44.set_yscale('log')
    ax44.set_ylabel(r'$ \| (\delta r^{\mathcal{O}}, \delta \dot{r}^{\mathcal{O}}) \|_2$', fontsize =14)
    ax44.set_xlabel("Time $(s)$", fontsize =14)
    leg = ax44.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax44.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgTransError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig45, ax45 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax45.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorRot'], color = myColor[i], label = lbl)
        i+=1
    # ax44.set_yscale('log')
    ax45.set_ylabel(r'$ \| (\delta q_{\mathcal{D}}^{\mathcal{O}} - q^I, \delta \omega_{\mathcal{D}}^{\mathcal{O}}) \|_2$', fontsize =14)
    ax45.set_xlabel("Time $(s)$", fontsize =14)
    leg = ax45.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax45.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgRotError.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig46, ax46 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax46.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorThrust'], color = myColor[i], label = lbl)
        i+=1
    # ax46.set_yscale('log')
    ax46.set_ylabel(r'$ \| F^{\mathcal{D}}_d \|_2$', fontsize =14)
    ax46.set_xlabel("Time $(s)$", fontsize =14)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(2,4))
    leg = ax46.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax46.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgErrorThrust.eps', format = 'eps', bbox_inches='tight')
    plt.show()

    i = 0
    fig47, ax47 = plt.subplots()
    plt.style.use('default')
    for key in itersDict:
        lbl = key
        if lbl == 10000 or lbl == '10000':
            lbl = 'Optimal'
        ax47.plot(itersDict[key]['avgTimescale'], itersDict[key]['avgErrorTau'], color = myColor[i], label = lbl)
        i+=1
    # ax47.set_yscale('log')
    ax47.set_ylabel(r'$ \| \tau^{D} \|_2$', fontsize =14)
    ax47.set_xlabel("Time $(s)$", fontsize =14)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(2,4))
    leg = ax47.legend(fontsize =10,loc = "upper right", ncol = 3)
    leg.set_title(r'$j_{\max}$',prop={'size':12})
    ax47.grid()
    if savePlots == True:
        plt.savefig('/home/gbehrendt/CLionProjects/Satellite/Figures/avgErrorTau.eps', format = 'eps', bbox_inches='tight')
    plt.show()
    