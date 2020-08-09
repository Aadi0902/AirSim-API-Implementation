# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 08:32:02 2020

@author: amk170930
"""


import airsim
import setup_path 
import time
from airsim import Vector3r, Quaternionr, Pose
from airsim.utils import to_quaternion
from datetime import datetime

class NHtestingAK:
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()
    car_state = client.getCarState()
    VehicleClient = airsim.VehicleClient()
    
    pos = car_state.kinematics_estimated.position
    print(pos)
    
    #Let it set
    time.sleep(3)
    
    #Clear previous plot
    VehicleClient.simFlushPersistentMarkers()
    
    #Plot the path
    #Straight line
    VehicleClient.simPlotArrows(points_start = [Vector3r(pos.x_val,pos.y_val,0)],
                                                points_end = [Vector3r(68,0,0)],
                            color_rgba = [1.0, 0.0, 0.0, 1.0], arrow_size = 10, thickness = 10, is_persistent = True)
    
    
    car_state = client.getCarState()
    pos = car_state.kinematics_estimated.position
    
     # go forward
    car_controls.throttle = 0.75
    car_controls.steering = 0
    client.setCarControls(car_controls)
    print("Go Forward")
    # time.sleep(9.5)   # let car drive a bit
    while pos.x_val < 68:
        car_state = client.getCarState()
        pos = car_state.kinematics_estimated.position
    
    car_state = client.getCarState()
    pos = car_state.kinematics_estimated.position
    print(pos)
    
    # Go forward + steer left
    car_controls.throttle = 0.5
    car_controls.steering = -1
    client.setCarControls(car_controls)
    print("Go Forward, steer left")
    time.sleep(1.463)   # let car drive a bit
    
    car_state = client.getCarState()
    pos = car_state.kinematics_estimated.position
    print(pos)
    
    #Get current position of the car and plot accordingly
    car_state = client.getCarState()
    pos = car_state.kinematics_estimated.position
    

    #Clear previous plot
    VehicleClient.simFlushPersistentMarkers()
    
     #Left line
    VehicleClient.simPlotArrows(points_start = [Vector3r(pos.x_val,pos.y_val,0)],
                                                points_end = [Vector3r(pos.x_val,-120,0)],
                            color_rgba = [1.0, 0.0, 0.0, 1.0], arrow_size = 10, thickness = 10, is_persistent = True)
    
    # go forward
    car_controls.throttle = 0.75
    car_controls.steering = 0
    client.setCarControls(car_controls)
    print("Go Forward")
    time.sleep(7)   # let car drive a bit
    
    # apply brakes
    car_controls.brake = 1
    client.setCarControls(car_controls)
    print("Apply brakes")
    time.sleep(3)   # let car drive a bit
    car_controls.brake = 0 #remove brake
    
    car_state = client.getCarState()
    pos = car_state.kinematics_estimated.position
    print(pos)
    
    #restore to original state
    client.reset()

    client.enableApiControl(False)