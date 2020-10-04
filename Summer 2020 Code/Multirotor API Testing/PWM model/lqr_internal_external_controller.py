# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:19:10 2020

@author: vxr131730
"""
##############################################################################
##############################################################################
###########################  IMPORT LIBRARIES ################################
##############################################################################
##############################################################################

import numpy as np
import airsim
import time
from scipy import signal
import matrixmath
from numpy import linalg as la
from scipy.spatial.transform import Rotation as scipy_rotation


##############################################################################
##############################################################################
########################### DEFINE THE GLOBAL VARIABLES ######################
##############################################################################
##############################################################################

RADS2RPM = 60/(2*np.pi)
RAD2DEG  = 180/np.pi

##############################################################################
##############################################################################
###########################  DEFINE THE FUNCTIONS  ###########################
##############################################################################
##############################################################################

def not_reached(pt1, pt2, dist):
    if np.linalg.norm(pt1[0:3] - pt2[0:3]) > dist:
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

# System parameters
pwmHover            = 0.59375                                                # PWM hover constant
max_rpm             = 6396.667
n_motor             = 4                                                                    #Number of motors
l                   = 0.2275                                                   # Arm length [m]
l_arm_x             = l*np.sqrt(2)/2 #Arm length in X axis [m]
l_arm_y             = l*np.sqrt(2)/2 #Arm length in Y axis [m]
l_arm_z             = 0.025 #Arm length in Z axis [m]
l_box_x             = 0.18 #Central body length in X axis [m]
l_box_y             = 0.11 #Central body length in Y axis [m]
l_box_z             = 0.04 #Central body length in Z axis [m]
l_feet              = 0.1393 # Feet length in Z axis [m]
Diam                = 0.2286 #Rotor diameter [m]
Rad                 = Diam/2 #Rotor radius [m]

# Mass Parameters
m                   = 1# Mass [kg]
m_motor             = 0.055# Motor mass [kg]
m_box               = m-n_motor*m_motor # Central body mass [kg]
g                   = 9.8 # Gravity [m/s**2]
Ixx                 = m_box/12*(l_box_y**2+l_box_z**2)+(l_arm_y**2+l_arm_z**2)*m_motor*n_motor
# Inertia in X axis [kg m**2]
Iyy                 = m_box/12*(l_box_x**2+l_box_z**2)+(l_arm_x**2+l_arm_z**2)*m_motor*n_motor
# Inertia in Y axis [kg m**2]
Izz                 = m_box/12*(l_box_x**2+l_box_y**2)+(l_arm_x**2+l_arm_y**2)*m_motor*n_motor
# Inertia in Z axis [kg m**2]
Ir                  = 2.03e-5; # Rotor inertia around spinning axis [kg m**2]

# Motor Parameters
max_rpm             = 6396.667 # Rotor max RPM

max_omega           = max_rpm/RADS2RPM #Rotor max angular velocity [rad/s]
Tm                  = 0.005 # Motor low pass filter

# Aerodynamics Parameters
CT                  = 0.109919 # Traction coefficient [-]
CP                  = 0.040164 # Moment coefficient [-]
rho                 = 1.225 # Air density [kg/m**3]
k1                  = CT*rho*Diam**4
b1                  = CP*rho*Diam**5/(2*np.pi)
Tmax                = k1*(max_rpm/60)**2 #Max traction [N]
Qmax                = b1*(max_rpm/60)**2 # Max moment [Nm]
k                   = Tmax/(max_omega**2) # Traction coefficient
b                   = Qmax/(max_omega**2) # Moment coefficient
c                   = (0.04-0.0035) # Lumped Drag constant
KB                  = 2 # Fountain effect (:2)

# Contact Parameters
max_disp_a          = 0.001# Max displacement in contact [m]
n_a                 = 4 #Number of contacts [-]
xi                  = 0.95 # Relative damping [-]
ka                  = m*g/(n_a*max_disp_a) #Contact stiffness [N/m]
ca                  = 2*m*np.sqrt(ka/m)*xi*1/np.sqrt(n_a) #Contact damping [Ns/m]
mua                 = 0.5 #Coulomb friction coefficient [-]


k1                  = CT*rho*Diam**4
b1                  = CP*rho*Diam**5/(2*np.pi)
Tmax                = k1*(max_rpm/60)**2 # Max
Tmax                = k1*(max_rpm/60)**2# Max traction [N]
Qmax                = b1*(max_rpm/60)**2# Max moment [Nm]
k                   = Tmax/(max_omega**2)# Traction coefficient
b                   = Qmax/(max_omega**2)# Moment coefficient
c                   = (0.04-0.0035) # Lumped Drag constant
KB                  = 2
sq_ctrl_hover      = pwmHover*(max_omega**2)*((2*np.pi)**2) 


def External_Loop(Ts = 0.1, max_angular_vel = 6396.667 * 2* np.pi/ 60):

    # Get hover control action
    u0 = np.array([[m*g],[0],[0],[0]])
    
    
    # System dynamics matrix: states order [x y z phi theta psi u v w p q r]
    A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],                 
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    
    # Control Input matrix
    B = np.array([[0,   0],
                  [0,   0],
                  [0,   g],
                  [-g,  0]])     
    
    # Define the output matrices
    C = np.identity(4)
    D = np.zeros((4,2))
   
    # Gravity compensation component for control input to be added
    u_bar = np.array([[sq_ctrl_hover],
                      [sq_ctrl_hover],
                      [sq_ctrl_hover],
                      [sq_ctrl_hover]])          
   
    # Define the state and control penalty matrices    
    Q = np.diag([1,1,0,0])
    R = 0.01*np.identity(2)
   
    # Form the contnuous state space model
    sys  = signal.lti(A,B,C,D)
    
    # Discretize it using the time step Ts 
    sysd = signal.cont2discrete((sys.A,sys.B,sys.C,sys.D),Ts)
    
    # Compute the LQR gain matrix
    _, K_ext = matrixmath.dare_gain(sysd[0],sysd[1], Q, R)

    return K_ext

##############################################################################
##############################################################################    

def Internal_Loop(Ts = 0.1, max_angular_vel = 6396.667 * 2* np.pi/ 60):

    
    # Get hover control action
    u0 = np.array([[m*g],[0],[0],[0]])
    
    
    # System dynamics matrix: states order [x y z phi theta psi u v w p q r]
    A = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0],                  
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0]])
    
    # Matrix to convert [w1**2 w2**2 w3**2 w4**2] to [thrust, tx,ty,tz]
    Gamma = np.array([[  k,   k,   k,   k],
                      [ -np.sqrt(2)*l*k/2,  np.sqrt(2)*l*k/2, np.sqrt(2)*l*k/2, -np.sqrt(2)*l*k/2],
                      [ -np.sqrt(2)*l*k/2, np.sqrt(2)*l*k/2,  -np.sqrt(2)*l*k/2, np.sqrt(2)*l*k/2],
                      [  -b,   -b,    b,   b]])
    
    # Control Input matrix
    B = np.array([[0,   0,   0,   0,  1/m,   0 ,   0 ,     0],
                  [0,   0,   0,   0,   0 , 1/Ixx,   0 ,     0],
                  [0,   0,   0,   0,   0 ,   0 , 1/Iyy,     0],
                  [0,   0,   0,   0,   0 ,   0 ,   0 ,   1/Izz]]).T 
    
    # Define the output matrices
    C = np.identity(8)
    D = np.zeros((8,4))       
   
    # Define the state and control penalty matrices    
    Q = np.diag([1,1,1,1,0.6,60,60,1e-5]) # np.identity(12)
    R = 0.01*np.identity(4)
   
    # Form the contnuous state space model
    sys  = signal.lti(A,B,C,D)
    
    # Discretize it using the time step Ts 
    sysd = signal.cont2discrete((sys.A,sys.B,sys.C,sys.D),Ts)
    
    # Compute the LQR gain matrix
    _, K_int = matrixmath.dare_gain(sysd[0],sysd[1], Q, R)

    return K_int, u0, Gamma, CT
    
##############################################################################
##############################################################################
###########################  MAIN FUNCTION  ##################################
##############################################################################
##############################################################################

def main(): 
    
    ###########################################################################
    ###################### Problem Definition Start ###########################
    ###########################################################################

    Ts              = 0.0003                      # Simulation Time Step 
    Time            = 12        
    length          = Time/Ts    
    max_angular_vel = 6396.667*2*np.pi/60       # Maximum angular velocity
    goal_error      = 0.1                       # Error to goal tolerance  
    k_const         = 0.000367717               # Thrust = kConst * w**2
    max_thrust      = 4.179446268               # Maximum thrust
    g               = 9.81                      # Gravity  
    mass            = 1.0                       # Mass
    pwmHover        = 0.59375 
    # Specify the goal state to be reached
    x_goal = np.array([[1.0],
                       [0.0],
                       [-2.0],
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
    print(state.kinematics_estimated.position)
    
    # Arm the drone
    print("arming the drone...")
    multirotorClient.armDisarm(True)
    # wind = airsim.Vector3r(0,10,0)
    # multirotorClient.simSetWind(wind)
    
    # Take off
    if state.landed_state == airsim.LandedState.Landed:
        print("taking off...")
        multirotorClient.takeoffAsync().join()
    else:
        multirotorClient.hoverAsync().join()            
        time.sleep(1)
    
    # Infer the drone position
    pos = state.kinematics_estimated.position
    pos = np.array([pos.x_val,pos.y_val,pos.z_val]) 
    
    # Get the LQR gain matrix
    K_int, u0, Gamma, cT = Internal_Loop(Ts, max_angular_vel)
    K_ext                = External_Loop(Ts, max_angular_vel)
    
    environ              = multirotorClient.simGetGroundTruthEnvironment()
    print('Env Air Density = ', environ.air_density)
    standard_air_density = 1.225
    air_density_ratio    = environ.air_density/standard_air_density
    print('Air Density Ratio', air_density_ratio)
    
    # Run the loop until goal is reached
    for i in range(int(length)):
        
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
        
        e_ext = error[[0,1,6,7],:]        
        u_ext = np.dot(K_ext, e_ext)
        # print(u_ext)
        e_int = error[[2, 3, 4,5,8,9,10,11],:]   
        e_int[[1,0]] += u_ext[[0,0]]
        e_int[[2,0]] += u_ext[[1,0]]
        u_int = np.dot(K_int, e_int)  + u0
        
        
        ##############################################################################
        ###########################  PWM CONTROL  ####################################
        # Convext [f,tx,ty,tz] to pwm
        pwm  = np.clip(( Gamma.T @ u_int)/(max_omega**2),0,1)
        
        # Apply the PWM to Airsim
        multirotorClient.moveByMotorPWMsAsync(pwm[0,0], pwm[1,0], pwm[2,0], pwm[3,0],Ts).join
        time.sleep(0.01)
        
        # Infer the state after applying the control
        state = multirotorClient.getMultirotorState()
        pos   = state.kinematics_estimated.position
        pos   = np.array([pos.x_val,pos.y_val,pos.z_val])
        print('Drone Position =', multirotorClient.getMultirotorState().kinematics_estimated.position)
    
    time.sleep(5)
    
    print("disarming...")
    multirotorClient.moveByMotorPWMsAsync(0, 0, 0, 0, 10).join
    multirotorClient.armDisarm(False)
        
    multirotorClient.enableApiControl(False)
    print("Simulation STOP.")

##############################################################################
##############################################################################

if __name__ == '__main__':
    main() 

##############################################################################
##############################################################################
###########################  END OF FILE  ####################################
##############################################################################
##############################################################################