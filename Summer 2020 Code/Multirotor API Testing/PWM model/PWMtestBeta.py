# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:02:17 2020

@author: amk170930
G"""


import numpy as np
from numpy import linalg as la
import airsim
import time
import gain_matrix_calculator as calK
from scipy.spatial.transform import Rotation as R

def bound_control(u, max_abs_roll_rate, max_abs_pitch_rate, max_abs_yaw_rate):
    max_vals = np.array([1.0, max_abs_roll_rate,max_abs_pitch_rate,max_abs_yaw_rate])
    min_vals = np.array([0.0,-max_abs_roll_rate,-max_abs_pitch_rate,-max_abs_yaw_rate])
    return np.maximum(np.minimum(u,max_vals),min_vals)

def not_reached(pt1, pt2, dist):
    if np.linalg.norm(pt1[0:3] - pt2[0:3]) > dist:
        return True
    else:
        return False
    
class PWMtest:
    
    def main(self):
         multirotorClient = airsim.MultirotorClient()
         multirotorClient.confirmConnection()
         multirotorClient.enableApiControl(True)
         
         pwm = 0.59375
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
        
        # Maximum angular velocity convert to rps
        max_angular_vel = 6396.667  / 60
        max_thrust = 4.179446268;
        k_const = 0.00036771704516278653
        dist_check = 5
        #Final state
        x_bar = np.array([[1.0],
                          [0.0],
                          [4.0],
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
        
        
        print("Controls start")
        
        state = multirotorClient.getMultirotorState()
        pos = state.kinematics_estimated.position
        pos = np.array([pos.x_val,pos.y_val,pos.z_val])
        #while not_reached((x_bar[0],x_bar[1],x_bar[2]), pos, dist_check):
        for ind in range(10000):
            # Get state of the multiorotor
            state = multirotorClient.getMultirotorState()
            state = state.kinematics_estimated
            
            #initialState = state.position
            
           
            # Define rotation matrix (check2nd row sign)
#            rotationMatrix = np.array([[1, 0, 0],
#                                       [0, 1, 0],
#                                       [0, 0, 1]])
#            Rx = np.array([[1, 0, 0],
#                           [0, np.cos(np.pi), -np.sin(np.pi)],
#                           [0, np.sin(np.pi), np.cos(np.pi)]])
#        
#            Rz = np.array([[np.cos(-np.pi/4), -np.sin(-np.pi/4), 0],
#                          [np.sin(-np.pi/4), np.cos(-np.pi/4), 0],
#                          [0, 0, 1]])
#            #Rotate 45 degrees in z
#            R1 = np.array([[ np.sqrt(2)/2, -np.sqrt(2)/2, 0],
#                           [ np.sqrt(2)/2,  np.sqrt(2)/2, 0],
#                           [            0,            0,  1]])
#            # Rotate 180 in x
#            R2 = np.array([[1, 0 , 0],
#                           [0, -1, 0],
#                           [0, 0, -1]])
#            rotationMatrix = np.dot(Rx, Rz)
            rotationMatrix = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])
            #print(np.dot(rotationMatrix,pt))
            # Rotate 90 clockwise in z
            # R3 = np.array([ [ 0, 1, 0],
            #                 [-1, 0, 0],
            #                 [ 0, 0, 1]])

            #rotationMatrix = np.dot(R3, rotationMatrix)
            # Apply rotation to position
            position = np.array([[state.position.x_val], 
                                 [state.position.y_val], 
                                 [state.position.z_val]])   
            # Apply rotation to linear velocity
            linear_velocity =  np.array([[state.linear_velocity.x_val], 
                                         [state.linear_velocity.y_val], 
                                         [state.linear_velocity.z_val]])
            
            # Convert quaternion to rotation object
            q = R.from_quat([state.orientation.x_val,
                             state.orientation.y_val,
                             state.orientation.z_val,
                             state.orientation.w_val])
            
            # Apply rotation to euler angles and convert them back to euler
            r = (R.from_matrix(np.dot(rotationMatrix, q.as_matrix())))
            # Debugged 'zyx' to 'xyz' to match https://www.andre-gaschler.com/rotationconverter/
            orientation = r.as_euler('xyz')
            
            # r = np.dot(rotationMatrix, np.array([[state.angular_velocity.x_val],
            #                                      [state.angular_velocity.y_val],
            #                                      [state.angular_velocity.z_val]]))
            
            # Apply rotation matrix to angular velocity
            # Page 21 - https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2016/RD2016script.pdf
            angularVelMatrix = np.array([[state.angular_velocity.x_val],
                                         [state.angular_velocity.y_val],
                                         [state.angular_velocity.z_val]])

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
            
            K, u_bar, Gamma = calK.gainMatrix(Ts,max_angular_vel, rotationMatrix)
            # Compute u
            #print(u_bar)
            u_ang_vel = u_bar + np.dot(K, x_bar - x)
            # thrust = (k_const * u_ang_vel)
            # thrust = np.true_divide(thrust,(np.pi*2)**2)
            u_airsim = np.dot(Gamma,u_ang_vel)
            #print(u)
            #Controller frame transformation
            # Sabatino's orientation
            # 0 - CW +ve y
            # 1 - CCW +ve x
            # 2 - CW -ve y
            # 3 - CCW -ve x
            
            #Airsim current orientation (after rotation):
            # 0 - CCW +ve x
            # 1 - CCW -ve x
            # 2 - CW -ve y
            # 3 - CW +ve y
            # Q = np.array([[0.0, 1.0, 0.0, 0.0],
            #               [0.0, 0.0, 0.0, 1.0],
            #               [0.0, 0.0, 1.0, 0.0],
            #               [1.0, 0.0, 0.0, 0.0]])
            # Q = np.array([[0.5, 0.0, 0.0, 0.5],
            #               [0.0, 0.5, 0.5, 0.0],
            #               [0.5, 0.5, 0.0, 0.0],
            #               [0.0, 0.0, 0.5, 0.5]])
            
            # Q = np.array([[1.0, 0.0, 0.0, 0.0],
            #               [0, 1.0, 0.0, 0],
            #               [0.0, 0.0, 1.0, 0],
            #               [0, 0.0, 0.0, 1.0]])
            # u_45 = np.dot(Q, u)

            #u_45 = np.dot(la.inv(rotationMatrix),u_45)
            
            sq_ctrl = [max(u_ang_vel[0][0], 0.0),
                        max(u_ang_vel[1][0], 0.0),
                        max(u_ang_vel[2][0], 0.0),
                        max(u_ang_vel[3][0], 0.0)] # max is just in case norm of sq_ctrl_delta is too large (can be negative)
            # Convert rad/s^2 to (rps)^2 
            sq_ctrl = sq_ctrl/((2*np.pi)**2)
            pwm = (sq_ctrl*k_const)/max_thrust
            zer = [1,1,1,1]
            pwm = min(pwm,zer)
#            pwm0 = min(sq_ctrl[0]/max_angular_vel**2,1.0)
#            pwm1 = min(sq_ctrl[1]/max_angular_vel**2,1.0)
#            pwm2 = min(sq_ctrl[2]/max_angular_vel**2,1.0)
#            pwm3 = min(sq_ctrl[3]/max_angular_vel**2,1.0)
           
            multirotorClient.moveByMotorPWMsAsync(pwm[0], pwm[1], pwm[2] , pwm[3],Ts).join()
            #multirotorClient.moveToPositionAsync(x_bar[0], x_bar[1], x_bar[2], 0, 1200,
                        #airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False,0), -1, 1).join()

            #multirotorClient.moveByMotorPWMsAsync(pwmHover, pwmHover, pwmHover, pwmHover, Ts).join()
#            u_airsim[0] = (0.59375/9.81)*u_airsim[0]
#            bound_control(u_airsim, 5, 5, 5)
#            rotmat_u = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
#            u_airsim[1:4] = rotmat_u @ u_airsim[1:4]
#            u_airsim = np.squeeze(u_airsim)
#            #if u_airsim[1] >5 or u_airsim[2] >5 or u_airsim[3] >5 or u_airsim[0]>1:
#            
#            multirotorClient.moveByAngleRatesThrottleAsync(u_airsim[1],
#                                                           u_airsim[2],
#                                                           u_airsim[3],
#                                                           u_airsim[0],
#                                                           Ts).join()
            state = multirotorClient.getMultirotorState()
            pos = state.kinematics_estimated.position
            pos = np.array([pos.x_val,pos.y_val,pos.z_val])
            print(multirotorClient.getMultirotorState().kinematics_estimated.position)
    # print(x_bar[0][0])
       # multirotorClient.moveToPositionAsync(x_bar[0][0], x_bar[1][0], -x_bar[2][0], 1.0).join()  

        state = multirotorClient.getMultirotorState()
        state = state.kinematics_estimated

        
       # print(state)
        time.sleep(10)
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
