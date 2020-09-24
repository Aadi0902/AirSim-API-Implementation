# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:02:17 2020
@author: amk170930
G"""

##############################################################################
##############################################################################
###########################  IMPORT LIBRARIES ####################################
##############################################################################
##############################################################################

import numpy as np
import csv
import airsim
import time
import matplotlib.pyplot as plt
import matrixmath
from scipy import signal
from numpy import linalg as la
from scipy.spatial.transform import Rotation as scipy_rotation


##############################################################################
##############################################################################
###########################  DEFINE THE FUNCTIONS  ###########################
##############################################################################
##############################################################################

def bound_control(u, max_abs_roll_rate, max_abs_pitch_rate, max_abs_yaw_rate):
    max_vals = np.array([1.0, max_abs_roll_rate,max_abs_pitch_rate,max_abs_yaw_rate])
    min_vals = np.array([0.0,-max_abs_roll_rate,-max_abs_pitch_rate,-max_abs_yaw_rate])
    return np.maximum(np.minimum(u,max_vals),min_vals)

##############################################################################
##############################################################################

def CheckGoalReach(pt1, pt2, dist):
    if np.linalg.norm(pt1[0:3] - pt2[0:3]) <= dist:
        return True
    else:
        return False
    
##############################################################################
##############################################################################

def GetRotationMatrix(axis_flag, angle):
    if axis_flag == 1: # x axis
        rotationMatrix = np.array([[1, 0, 0],
                                   [0, np.cos(angle), -np.sin(angle)],
                                   [0, np.sin(angle), np.cos(angle)]])
        
    if axis_flag == 2: # y axis
        rotationMatrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                   [0, 1, 0],
                                   [-np.sin(angle), 0, np.cos(angle)]])
        
    if axis_flag == 3: # z axis
        rotationMatrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                   [np.sin(angle), np.cos(angle), 0],
                                   [0, 0, 1]])
    
    return rotationMatrix

    
##############################################################################
##############################################################################    

def gain_matrix_calculator(Ts = 0.1, max_angular_vel = 6396.667 * 2* np.pi/ 60):

    mass               = 1 - 0.055*4                                            # Mass of the quadrotor
    I                  = np.diag([0.00234817178, 0.00366767193, 0.00573909376]) # Inertial Matrix
    d                  = 0.2275                                                 # Center to rotor distance
    cT                 = 0.109919                                               # Torque coefficient
    cQ                 = 0.040164                                               # Drag coefficient
    air_density        = 1.225                                                  # Air density
    propeller_diameter = 0.2286                                                 # Propeller Diameter
    cT                 = cT*air_density*(propeller_diameter**4)*((2*np.pi)**2)  # Thrust coefficient
    cQ                 = 0.040164*(propeller_diameter**5)*air_density/(2*np.pi) # Drag coefficient
    g                  = 9.81                                                   # Gravity                             
    maxThrust          = 4.179446268                                            # Maximum thrust
    maxtTorque         = 0.055562                                               # Maximum torque
    pwmHover           = 0.59375                                                # PWM hover constant
    kConst             = 0.000367717                                            # Thrust = kConst * w^2
    sq_ctrl_hover      = pwmHover*(max_angular_vel**2)                          # sq_ctrl_hover
    
    # System dynamics matrix: states order [x y z phi theta psi u v w p q r]
    A = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0,-g, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, g, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    # Matrix to convert [w1^2 w2^2 w3^2 w4^2] to [thrust, tx,ty,tz]
    Gamma = np.array([[  cT,   cT,   cT,   cT],
                      [ -d*cT,  d*cT, d*cT, -d*cT],
                      [ d*cT, -d*cT,  d*cT, -d*cT],
                      [  cQ,   cQ,    -cQ,   -cQ]])
    
    # Control Input matrix
    B = np.array([[0,   0,   0,   0,   0,   0,   0,   0,  1/mass,   0 ,   0 ,     0],
                  [0,   0,   0,   0,   0,   0,   0,   0,   0 , 1/I[0,0],   0 ,     0],
                  [0,   0,   0,   0,   0,   0,   0,   0,   0 ,   0 , 1/I[1,1],     0],
                  [0,   0,   0,   0,   0,   0,   0,   0,   0 ,   0 ,   0 ,   1/I[2,2]]]) 
    
    # Change B matrix to have [w1^2 w2^2 w3^2 w4^2] as control inputs
    B = B.T @ Gamma
    
    # Define the output matrices
    C = np.identity(12)
    D = np.zeros((12,4))
   
    # Gravity compensation component for control input to be added
    u_bar = np.array([[sq_ctrl_hover],
                      [sq_ctrl_hover],
                      [sq_ctrl_hover],
                      [sq_ctrl_hover]])          
   
    # Define the state and control penalty matrices    
    Q = np.identity(12)
    R = np.identity(4)
   
    # Form the contnuous state space model
    sys  = signal.lti(A,B,C,D)
    
    # Discretize it using the time step Ts 
    sysd = signal.cont2discrete((sys.A,sys.B,sys.C,sys.D),Ts)
    
    # Compute the LQR gain matrix
    _, K = matrixmath.dare_gain(sysd[0],sysd[1], Q, R)

    return K, u_bar, Gamma
    
##############################################################################
##############################################################################
###########################  MAIN FUNCTION  ##################################
##############################################################################
##############################################################################

def main(): 
    
    ###########################################################################
    ###################### Problem Definition Start ###########################
    ###########################################################################

    Ts              = 0.001              # Simulation Time Step    
    max_angular_vel = 6396.667*2*np.pi/60 # Maximum angular velocity
    goal_error      = 0.3                 # Distance tolerance to the goal
    k_const         = 0.000367717         # Thrust = kConst * w^2
    max_thrust      = 4.179446268         # Maximum thrust
    g               = 9.81                # Gravity  
    mass            = 1.0                 # Mass
    
    # Specify the goal state to be reached
    x_goal = np.array([[-1.0],
                       [-10.0],
                       [-5.0],
                       [0.0],
                       [0.0],
                       [0.0],
                       [0.0],
                       [0.0],
                       [0.0],
                       [0.0],
                       [0.0],
                       [0.0]])

    # Setup the airsim multirotor multirotorClient
    multirotorClient = airsim.MultirotorClient()
    
    # Establish the connection to it and enable the control
    multirotorClient.confirmConnection()
    multirotorClient.enableApiControl(True)
    
    # Get the drone state
    state = multirotorClient.getMultirotorState()
    
    # Arm the drone
    print("Arming the drone...")
    multirotorClient.armDisarm(True)
    
    # Take off commented
    if state.landed_state == airsim.LandedState.Landed:
        print("Taking off...")
        multirotorClient.takeoffAsync().join()
    else:
        multirotorClient.hoverAsync().join()            
        time.sleep(1)
    
    # Infer the drone position
    pos = state.kinematics_estimated.position
    pos = np.round_(np.array([pos.x_val,pos.y_val,pos.z_val]), 2)
    print('Drone Start Positions: x =', pos[0], 'y =', pos[1], 'z =', pos[2]) 
    
    # Create new file for soring values
    file = open('Expt1.csv', 'w+', newline ='') 
    
    # Store current state
    totalTS      = 0
    storeTS      = np.empty((1,0))
    storeState   = np.empty((12,0))
    storeGoal    = np.empty((12,0))
    storeError   = np.empty((12,0))
    storeUAngVel = np.empty((4,0))
    storeUBar    = np.empty((4,0))
    storeK       = np.empty((4,0))
    storePWM     = np.empty((4,0))
    
    # Store the history of states
    pos_x_hist = [pos[0]]
    pos_y_hist = [pos[1]]
    pos_z_hist = [pos[2]]
    
    # Get the environmental air density
    environ_air_sensity  = multirotorClient.simGetGroundTruthEnvironment().air_density
    standard_air_density = 1.225
    air_density_ratio    = environ_air_sensity/standard_air_density
    
    # Get the LQR gain matrix
    K, u_bar, Gamma = gain_matrix_calculator(Ts, max_angular_vel)
    
    # Run the loop until goal is reached
    while True:
        
        # Check if goal is reached, if yes - stop
        if CheckGoalReach((x_goal[0,0],x_goal[1,0],x_goal[2,0]), pos, goal_error):
            
            # Land Smoothly    
            print("Reached Goal - Landing smoothly")
            
            multirotorClient.hoverAsync().join()
            time.sleep(1)
            multirotorClient.landAsync().join()
            
            print("Drone disarmed")    
            multirotorClient.armDisarm(False)
                
            multirotorClient.enableApiControl(False)
            print("Simulation STOP.")
            
            # Break the Simulation Loop
            break
        
        # Get state of the multirotor
        state = multirotorClient.getMultirotorState()
        state = state.kinematics_estimated
        
        # Get the rotation matrices for 180 deg rotation in x axis and -45 deg rotation in z axis
        Rx = GetRotationMatrix(1, np.pi)
        Rz = GetRotationMatrix(3, -np.pi/4)
    
        # Get the desired rotation
        rotationMatrix = np.identity(3) # np.dot(Rx, Rz)  # np.identity(3)
        
        # Get the position of drone before rotation
        pos_before_rot     = np.array([[state.position.x_val], 
                                       [state.position.y_val], 
                                       [state.position.z_val]])
        # Get the linear velocity of drone before rotation
        lin_vel_before_rot = np.array([[state.linear_velocity.x_val], 
                                       [state.linear_velocity.y_val], 
                                       [state.linear_velocity.z_val]])
        # Get the angular velocity of drone before rotation
        ang_vel_before_rot = np.array([[state.angular_velocity.x_val],
                                       [state.angular_velocity.y_val],
                                       [state.angular_velocity.z_val]])
        
        # Apply rotation to position
        position = np.dot(rotationMatrix, pos_before_rot)   
        
        # Apply rotation to linear velocity
        linear_velocity =  np.dot(rotationMatrix, lin_vel_before_rot)
        
        # Convert quaternion to rotation object
        q = scipy_rotation.from_quat([state.orientation.x_val,
                                      state.orientation.y_val,
                                      state.orientation.z_val,
                                      state.orientation.w_val])
        
        # Apply rotation to euler angles and convert them back to euler
        r = (scipy_rotation.from_matrix(np.dot(rotationMatrix, q.as_matrix())))
        
        # Debugged 'zyx' to 'xyz' to match https://www.andre-gaschler.com/rotationconverter/
        orientation = r.as_euler('xyz')
        
        # Apply rotation matrix to angular velocity
        # Page 21 - https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2016/RD2016script.pdf
        angularVelMatrix = np.dot(rotationMatrix.T, ang_vel_before_rot)

        # Form the state vector
        x = np.array([[position[0,0]],
                      [position[1,0]],
                      [position[2,0]],
                      [orientation[0]],
                      [orientation[1]],
                      [orientation[2]],
                      [linear_velocity[0,0]],
                      [linear_velocity[1,0]],
                      [linear_velocity[2,0]],
                      [angularVelMatrix[0,0]],
                      [angularVelMatrix[1,0]],
                      [angularVelMatrix[2,0]]])
        
        # Define the error to goal
        error = x - x_goal        
        # print('error=', error)
        
        # Compute the desired control inputs - [w1^2 w2^2 w3^2 w4^2] 
        u_ang_vel = np.dot(K, error)        
                
        ##############################################################################
        ###########################  PWM CONTROL  ####################################
            
        # Add the gravity compensation component to all the control inputs
        u_ang_vel += u_bar       
        
        # Remove negative values
        # for kk in range(u_ang_vel.shape[0]):
        #     if u_ang_vel[kk] < 0:
        #         u_ang_vel[kk] = 0.0
        
        # Convert u_ang_vel in rad/s^2 to sq_ctrl in (rps)^2 
        # u_ang_vel = u_ang_vel/((2*np.pi)**2)
        
        # Compute the required PWM control inputs for the four rotors
        pwm = u_ang_vel/(air_density_ratio*max_angular_vel**2)     
        
        for kk in range(pwm.shape[0]):
            if pwm[kk] > 1:
                pwm[kk] = 1        

        # Apply the PWM to Airsim
        multirotorClient.moveByMotorPWMsAsync(pwm[0,0], pwm[1,0], pwm[2,0], pwm[3,0], Ts).join()
        
        # CSV code
        totalTS      = totalTS + Ts        
        storeTS      = np.append(storeTS,np.array([[totalTS]]),axis=1)
        storeState   = np.append(storeState,x,axis=1)
        storeGoal    = np.append(storeGoal,x_goal,axis=1)
        storeError   = np.append(storeError,error,axis=1)
        storeUAngVel = np.append(storeUAngVel,u_ang_vel,axis=1)
        storeUBar    = np.append(storeUBar,u_bar,axis=1)
        storeK       = np.append(storeK,K,axis=1)
        storePWM     = np.append(storePWM,pwm,axis=1)
        
        # Infer the state after applying the control
        state = multirotorClient.getMultirotorState()
        pos   = state.kinematics_estimated.position
        pos   = np.round_(np.array([pos.x_val,pos.y_val,pos.z_val]), 2)
        print('Drone Positions: x =', pos[0], 'y =', pos[1], 'z =', pos[2])
        
        # Store the drone position history
        pos_x_hist.append(pos[0])
        pos_y_hist.append(pos[1])
        pos_z_hist.append(pos[2])
    

    # Prepare for plotting
    pos_z_hist  = [-x for x in pos_z_hist]
    goal_x_hist = [x_goal[0,0]]*len(pos_x_hist)  
    goal_y_hist = [x_goal[1,0]]*len(pos_y_hist)
    goal_z_hist = [-x_goal[2,0]]*len(pos_z_hist)
    
    # Plot the history of states
    tme = np.arange(0,totalTS, Ts).tolist()
    ax1 = plt.subplot(311)    
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)
    ax1.plot(tme, pos_x_hist)        
    ax2.plot(tme, pos_y_hist)        
    ax3.plot(tme, pos_z_hist)
    ax1.plot(tme, goal_x_hist)
    ax2.plot(tme, goal_y_hist)
    ax3.plot(tme, goal_z_hist)
    plt.show() 

##############################################################################
##############################################################################

if __name__ == '__main__':
    main() 

##############################################################################
##############################################################################
###########################  END OF FILE  ####################################
##############################################################################
##############################################################################