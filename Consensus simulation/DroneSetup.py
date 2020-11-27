# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 03:27:16 2020

@author: amk170930
"""


import numpy as np
import airsim
import time

client = airsim.MultirotorClient()
client.confirmConnection()

drones = np.array(['Drone1','Drone2','Drone3','Drone4','Drone5'])

for drone in drones:
    # Enable control
    client.enableApiControl(True, drone) 
    
    # Arm the drones
    print("arming the drone...")
    client.armDisarm(True, drone)
    client.enableApiControl(True, drone)
    
    client.confirmConnection()
    # Try taking off the multirotor to a deafult height of z = -3
    state = client.getMultirotorState(drone)
    if state.landed_state == airsim.LandedState.Landed:
        print("taking off...")
        client.takeoffAsync(vehicle_name = drone)
    else:
        client.hoverAsync(vehicle_name = drone)
    
    z = -5
    print("make sure we are hovering at %.1f meters..." %(-z))
    client.moveToZAsync(z, 1, vehicle_name = drone)




time.sleep(100)
