# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 12:26:45 2020

@author: amk170930
"""


import pickle
import numpy as np
from squaternion import Quaternion
import airsim

 # #Setup airsim multirotor multirotorClient
multirotorClient = airsim.MultirotorClient()
multirotorClient.confirmConnection()
multirotorClient.enableApiControl(True)

multirotorClient.moveToPositionAsync(x, y, z, velocity).join
state = multirotorClient.getMultirotorState()
multirotorClient.moveByMotorPWMsAsync(1, 1, 1, 1, 10).join()
multirotorClient.moveByMotorPWMsAsync(0.5, 1, 0.5, 1, 5).join()
#state = multirotorClient.getMultirotorState()
# print(state.kinematics_estimated)
# u_c = pickle.load( open( "control_inputs.p", "rb" ) )
# u_c = np.squeeze(u_c)
# print('Total TimeSteps =', u_c.shape[0])
# for i in range(10): 
#     print(u_c[i])  # will give you the control inputs for the 4 rotors at initial time step (t = 0)
