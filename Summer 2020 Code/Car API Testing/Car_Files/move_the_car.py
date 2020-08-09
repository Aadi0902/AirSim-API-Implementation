import airsim
import cv2
import numpy as np
import os
import setup_path 
import pickle
import time
import math


# Load the control commands from pickled data
filename  = 'CarControlList'
infile = open(filename,'rb')
uMPC = pickle.load(infile)
infile.close()



# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

# Get the position of the car
car_position = client.simGetVehiclePose().position
print("At the start, the car is at (%d, %d)" % (car_position.x_val, car_position.y_val))

# Set the position of the car
client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(car_position.x_val - 100, 
                                                      car_position.y_val, 
                                                      car_position.y_val), 
                                      airsim.to_quaternion(0, 0, np.pi/2)), 
                          True)

car_position = client.simGetVehiclePose().position
print("The new position of car is (%d, %d)" % (car_position.x_val, car_position.y_val))

# Get to know the quaternion based orientation of the car
car_state = client.getCarState()
print(car_state)
car_orientation = car_state.kinematics_estimated.orientation
# Get the quaternion values
q0 = car_orientation.w_val
q1 = car_orientation.x_val
q2 = car_orientation.y_val
q3 = car_orientation.z_val
# Calculate the yaw heading of the car
# Link: https://en.m.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
car_heading = math.atan2(2*(q0*q3 + q1*q2), 1-2*(q2**2 + q3**2))
print('Car Heading is ', car_heading, 'rad')

for k in range(len(uMPC)):
    # Get the controls from the list
    u = uMPC[k]
    print('Commanded Throttle = ', u[0], 'Commanded Steer = ', u[1])  
    
    # car_controls.throttle = u[0]
    # car_controls.steering = u[1]
    # client.setCarControls(car_controls)
    # time.sleep(1)   # let car drive a bit
    
    if k == 0:
        desiredThrottle = u[0]
        desiredSteer    = u[1]
    
    if k > 0:
        u_curr = uMPC[k]
        u_prev = uMPC[k-1]
        commandedThrottle = u_curr[0]
        previousThrottle  = u_prev[0]
        commandedSteer    = u_curr[1]
        previousSteer     = u_prev[1]
        desiredThrottle   = commandedThrottle - previousThrottle        
        desiredSteer      = commandedSteer - previousSteer
    
    # If the desired throttle is positive drive
    if desiredThrottle > 0:        
        car_controls.throttle = desiredThrottle
        car_controls.steering = desiredSteer
        client.setCarControls(car_controls)
        time.sleep(2)   # let car drive a bit
    else:
        # If desired throttle is negative first engage the reverse gear, then drive        
        car_controls.is_manual_gear = True;
        car_controls.manual_gear = -1
        car_controls.throttle = desiredThrottle
        car_controls.steering = desiredSteer
        client.setCarControls(car_controls)
        time.sleep(2)   # let car drive a bit
        # Once driven disengage the reverse gear
        car_controls.is_manual_gear = False; # change back gear to auto
        car_controls.manual_gear = 0  
    
car_position = client.simGetVehiclePose().position
print("The final position of car is (%d, %d)" % (car_position.x_val, car_position.y_val))

#restore to original state
client.reset()

client.enableApiControl(False)


            
