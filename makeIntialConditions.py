#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:02:02 2023

@author: gabe
"""
import csv
import random
import numpy as np


savePath = "/home/gbehrendt/CLionProjects/Satellite/InitialConditions300.csv"
numConditions = 300

pMax = 1.5
pMin = -pMax
vMax = 1e-3
vMin = -vMax
qMax = 1
qMin = 0
wMax = 2e-3
wMin = -wMax

tPeriod = 92.68 * 60
n = -2*np.pi/tPeriod

initialConditions = []
for i in range(numConditions):
    x = random.uniform(pMin,pMax)
    y = random.uniform(pMin,pMax)
    z = random.uniform(pMin,pMax)
    
    dx = random.uniform(vMin,vMax)
    dy = random.uniform(vMin,vMax)
    dz = random.uniform(vMin,vMax)
    
    sq = random.uniform(qMin,qMax)
    v1 = random.uniform(qMin,qMax)
    v2 = random.uniform(qMin,qMax)
    v3 = random.uniform(qMin,qMax)
    normq = np.sqrt(sq**2 + v1**2 + v2**2 + v3**2)
    
    # Normalize q
    sq = sq/normq
    v1 = v1/normq
    v2 = v2/normq
    v3 = v3/normq
    
    wx = random.uniform(wMin,wMax)
    wy = random.uniform(wMin,wMax)
    wz = random.uniform(wMin,wMax)
    
    # wx = 0
    # wy = 0
    # wz = n
    
    x0 = [i,x,y,z,dx,dy,dz,sq,v1,v2,v3,wx,wy,wz]
    initialConditions.append(x0)


# Opening a CSV file in write mode
with open(savePath, 'w', newline='') as file:
    # Step 4: Using csv.writer to write the list to the CSV file
    writer = csv.writer(file)
    writer.writerows(initialConditions) # Use writerows for nested list
    







