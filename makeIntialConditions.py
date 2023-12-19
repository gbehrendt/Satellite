#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 14:02:02 2023

@author: gabe
"""

import csv
import random
import numpy as np

# Step 2: Creating a Python list
my_list = ["t-shirts", "hoodies", "jeans"]

savePath = "/home/gabe/Satellite/intialConditions.csv"
numConditions = 10

pMax = 1
pMin = -pMax
vMax = 1e-3
vMin = -vMax
qMax = 1
qMin = 0
wMax = 5e-3
wMin = -wMax

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
    
    x0 = [i,x,y,z,dx,dy,dz,sq,v1,v2,v3,wx,wy,wz]
    initialConditions.append(x0)



# Step 2: Creating a Python list of lists
my_list = [["t-shirts", 9.99, 342], ["hoodies", 24.99, 118], ["jeans", 29.99, 612]]




# Step 3: Opening a CSV file in write mode
with open(savePath, 'w', newline='') as file:
    # Step 4: Using csv.writer to write the list to the CSV file
    writer = csv.writer(file)
    writer.writerows(initialConditions) # Use writerows for nested list
    

# Step 5: The file is automatically closed when exiting the 'with' bloc






