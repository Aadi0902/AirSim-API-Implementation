# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 07:38:31 2020

@author: amk170930
"""


import setup_path 
import airsim

import numpy as np
from airsim import Vector3r, Quaternionr, Pose
import os
import tempfile
import pprint
import cv2
import time
# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
VehicleClient = airsim.VehicleClient()
# Create file
f = open("Trajectory_dataXYZ.txt","a+")
f.write("\n\nRunning multirotor: \n")

#Get current multirotor state
state = client.getMultirotorState()
print(state)
pos = state.kinematics_estimated.position
# print(state)
startPosition = [pos.x_val, pos.y_val, pos.z_val]

i = 0

#Get initial time stamp
prevTime = state.timestamp
startTime = prevTime
# Run for 10 seconds
while i<10:
    
    # Move multirotor for 0.02 seconds
    client.moveByVelocityAsync(0, 20, -20,0.04).join()
    #client.simPause(True)

    # Get client state
    state = client.getMultirotorState()
    
    #print((state.timestamp-prevTime)/1000000000)
    
    # Get multirotor position
    pos = state.kinematics_estimated.position
    f.write("%f %f %f %f\n" %((state.timestamp-prevTime)/1000000000,
                                                  pos.x_val, pos.y_val, pos.z_val))
    endPosition = [pos.x_val, pos.y_val, pos.z_val]
    
    # client.simPlotArrows(points_start = [Vector3r(startPosition[0],startPosition[1],startPosition[2])],
    #                                             points_end = [Vector3r(endPosition[0],endPosition[1],endPosition[2])],
    #                         color_rgba = [1.0, 0.0, 0.0, 1.0],duration =20, arrow_size = 10, thickness = 5)
    
    startPosition = endPosition
    prevTime = state.timestamp
    i = i + 0.04
    #client.simPause(False)

    
currentTime = state.timestamp
print((currentTime-startTime)/1000000000)
  

VehicleClient.simFlushPersistentMarkers()  
f.close() 
client.armDisarm(False)
client.reset()