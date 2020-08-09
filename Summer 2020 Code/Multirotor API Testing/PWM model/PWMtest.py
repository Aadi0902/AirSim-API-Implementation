# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:02:17 2020

@author: amk170930
G"""


import numpy as np
import airsim
from airsim import Vector3r
import time
import xlrd
import control
import matrixmath
import gain_matrix_calculator as calK
from scipy import signal
from squaternion import Quaternion
import control.matlab
from scipy.spatial.transform import Rotation as R

class PWMtest:
    def main(self):
         multirotorClient = airsim.MultirotorClient()
         multirotorClient.confirmConnection()
         multirotorClient.enableApiControl(True)
         
         pwm = 0.6
         state = multirotorClient.getMultirotorState()
         initialTime = state.timestamp/1000000000
         
         for ind in range(5):
            print("Iteration: %d" %(ind)) 
            multirotorClient.moveByMotorPWMsAsync(pwm, pwm, pwm, pwm, 2).join()
            
         state = multirotorClient.getMultirotorState()
         FinalTime = state.timestamp/1000000000
         print("Time: %f" %(FinalTime - initialTime))
            
         print("Out")
         time.sleep(20)
         print("Hover")
         multirotorClient.hoverAsync().join()
         time.sleep(10)
     
class LQRtestPWM:
    def main(self):       
        #Time step
        Ts = 0.1
        
        # Maximum angular velocity
        max_angular_vel = 6393.667 * 2 * np.pi / 60
        
        #Final state
        x_bar = np.array([[0.0],
                          [0.0],
                          [10.0],
                          [0.0],
                          [0.0],
                          [0.0],
                          [0.0],
                          [0.0],
                          [0.0],
                          [0.0],
                          [0.0],
                          [0.0]])
        #Gain matrix


        # #Setup airsim multirotor multirotorClient
        multirotorClient = airsim.MultirotorClient()
        multirotorClient.confirmConnection()
        multirotorClient.enableApiControl(True)
        
        vehicleClient = airsim.VehicleClient()
        
        state = multirotorClient.getMultirotorState()
        print(state.kinematics_estimated.position)
        # Arm the drone
        print("arming the drone...")
        multirotorClient.armDisarm(True)
        
        if state.landed_state == airsim.LandedState.Landed:
            print("taking off...")
            multirotorClient.takeoffAsync().join()
        else:
                multirotorClient.hoverAsync().join()
                
                time.sleep(2)
        

        
        # Declare u matrix 4 x 1
        # u = [0,
        #      0,
        #      0,
        #      0]
        # pwm = np.array([0,
        #                 0,
        #                 0,
        #                 0])
        
        print("Controls start")
        #time.sleep(2)
        #multirotorClient.moveByMotorPWMsAsync(1, 1, 1, 1,3).join()
        #newX = [[],[],[],[],[],[],[],[],[],[],[],[]]
        # Start step loop
        for index in range(1000):
            # Re initilize u for every iteration
            # u = [0,
            #      0,
            #      0,
            #      0]

            # Get state of the multiorotor
            state = multirotorClient.getMultirotorState()
            state = state.kinematics_estimated
            
            initialState = state.position
            
            #Convert from quaternion to euler angle
            #euler = ls.quaternion_to_euler(state.orientation.x_val,state.orientation.y_val, state.orientation.z_val,state.orientation.w_val)
            
            # Convert quaternion to rotation object
            q = R.from_quat([state.orientation.x_val,
                             state.orientation.y_val,
                             state.orientation.z_val,
                             state.orientation.w_val])
            # Convert rotation object to euler angles
            #e = np.array(q.as_euler('zyx'))
            
            # Define rotation matrix (check2nd row sign)
            
            #Rotae 45 degrees in z
            R1 = np.array([[ np.sqrt(2)/2, np.sqrt(2)/2, 0],
                           [-np.sqrt(2)/2, np.sqrt(2)/2, 0],
                           [            0,            0, 1]])
            # Rotate 180 in x
            R2 = np.array([[1, 0 , 0],
                           [0, -1, 0],
                           [0, 0, -1]])
            # Rotate 90 in z
            R3 = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1]])
            # rotationMatrix =np.dot(np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
            #                                 [-np.sqrt(2)/2, -np.sqrt(2)/2, 0],
            #                                 [0, 0, -1]])
            
            rotationMatrix = np.dot(R1, R2)
            rotationMatrix = np.dot(R3, rotationMatrix)
            # Apply rotation to position
            position = np.dot(rotationMatrix,np.array([[state.position.x_val], 
                                                       [state.position.y_val], 
                                                       [state.position.z_val]]))   
            # Apply rotation to linear velocity
            linear_velocity =  np.dot(rotationMatrix,np.array([[state.linear_velocity.x_val], 
                                                               [state.linear_velocity.y_val], 
                                                               [state.linear_velocity.z_val]]))
            # Apply rotation to euler angles and convert them back to euler
            r = (R.from_matrix(np.dot(rotationMatrix, q.as_matrix())))
            orientation = r.as_euler('zyx')
            
            # r = np.dot(rotationMatrix, np.array([[state.angular_velocity.x_val],
            #                                      [state.angular_velocity.y_val],
            #                                      [state.angular_velocity.z_val]]))
            
            # Apply rotation matrix to angular velocity
            angularVelMatrix = np.dot(rotationMatrix, np.array([[state.angular_velocity.x_val],
                                         [state.angular_velocity.y_val],
                                         [state.angular_velocity.z_val]]))
            #R.from_euler(obj)
            #Store the current state of multirotor in x
            #e[2] = e[2] + np.pi if e[2]<=np.pi else e[2] - np.pi
            x = np.array([[position[0][0]],
                          [position[1][0]],
                          [position[2][0]],
                          [orientation[0]],
                          [orientation[1]],
                          [orientation[2]],
                          [linear_velocity[0][0]],
                          [linear_velocity[1][0]],
                          [linear_velocity[2][0]],
                          [angularVelMatrix[0][0]],
                          [angularVelMatrix[1][0]],
                          [angularVelMatrix[2][0]]])
            
            K, u_bar = calK.gainMatrix(Ts,max_angular_vel, rotationMatrix)
            # Compute u
            u = np.dot(K, x_bar-x) + u_bar
            
            #Controller frame transformation 
            Q = np.array([[0.5, 0.0, 0.0, 0.5],
                          [0, 0.5, 0.5, 0],
                          [0.0, 0.5, 0.0, 0.5],
                          [0.5, 0.0, 0.5, 0]])
            # Q = np.array([[1.0, 0.0, 0.0, 0.0],
            #               [0, 1.0, 0.0, 0],
            #               [0.0, 0.0, 1.0, 0],
            #               [0, 0.0, 0.0, 1.0]])
            u_45 = np.dot(Q, u)

            sq_ctrl = [max(u_45[0][0], 0.0),
                       max(u_45[1][0], 0.0),
                       max(u_45[2][0], 0.0),
                       max(u_45[3][0], 0.0)] # max is just in case norm of sq_ctrl_delta is too large (can be negative)

            pwm0 = min(sq_ctrl[0]/(max_angular_vel**2),1.0)
            pwm1 = min(sq_ctrl[1]/(max_angular_vel**2),1.0)
            pwm2 = min(sq_ctrl[2]/(max_angular_vel**2),1.0)
            pwm3 = min(sq_ctrl[3]/(max_angular_vel**2),1.0)
           
            multirotorClient.moveByMotorPWMsAsync(pwm0, pwm1, pwm2, pwm3,Ts).join()
            #multirotorClient.moveToPositionAsync(x_bar[0], x_bar[1], x_bar[2], 0, 1200,
                        #airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False,0), -1, 1).join()

            #multirotorClient.moveByMotorPWMsAsync(pwmHover, pwmHover, pwmHover, pwmHover, Ts).join()
        
       # print(x_bar[0][0])
       # multirotorClient.moveToPositionAsync(x_bar[0][0], x_bar[1][0], -x_bar[2][0], 1.0).join()  

        state = multirotorClient.getMultirotorState()
        state = state.kinematics_estimated

        
       # print(state)
        #time.sleep(10)
        print("Free fall")
        multirotorClient.moveByMotorPWMsAsync(0, 0, 0, 0, 10).join
        #time.sleep(10)
        
        print("disarming...")
        multirotorClient.armDisarm(False)
            
        multirotorClient.enableApiControl(False)
        print("done.")
            
    # def quaternion_to_euler(self,x, y, z, w):
    
            
    #         r = R.from_quat([x,y,z,w])
    #         r = r.as_euler('xyz')
    #         # import math
    #         # t0 = +2.0 * (w * x + y * z)
    #         # t1 = +1.0 - 2.0 * (x ** 2 + y ** y)
    #         # X = math.atan2(t0, t1)
    
    #         # t2 = +2.0 * (w * y - z * x)
    #         # t2 = +1.0 if t2 > +1.0 else t2
    #         # t2 = -1.0 if t2 < -1.0 else t2
    #         # Y = math.asin(t2)
    
    #         # t3 = +2.0 * (w * z + x * y)
    #         # t4 = +1.0 - 2.0 * (y * y + z * z)
    #         # Z = math.atan2(t3, t4)
    
    #         return r[0], r[1], r[2]

ls = LQRtestPWM()
ls.main()
