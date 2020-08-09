# -*- coding: utf-8 -*-
"""
Created on Sun May 31 03:44:40 2020

@author: amk170930
"""


import airsim
import numpy as np
import setup_path 
import time
from airsim import Vector3r, Quaternionr, Pose
from airsim.utils import to_quaternion
from datetime import datetime
class stepData:
    position = [[]]

class testingAK:
    startPosition = [0,0,0]
    
    clockSpeed =  1#From settings.json
    clockSpeedMult = 1/clockSpeed
    
    frequency = 10
    plotFreq = 1
    #Assumption: plotFreq<frequency
    timePeriod = 1/(frequency*clockSpeed)
    
    #Plots red arrows from start point to a given point
    def Plot(self,client = airsim.CarClient()):
        client.simPause(True)
        car_state = client.getCarState()
        VehicleClient = airsim.VehicleClient()
        pos = car_state.kinematics_estimated.position
        endPosition = [pos.x_val,pos.y_val,pos.z_val]
        #Plot red arrows persistently
        VehicleClient.simPlotArrows(points_start = [Vector3r(self.startPosition[0],self.startPosition[1],self.startPosition[2])],
                                                points_end = [Vector3r(endPosition[0],endPosition[1],endPosition[2])],
                            color_rgba = [1.0, 0.0, 0.0, 1.0], arrow_size = 10, thickness = 5, is_persistent = True)
        self.startPosition = endPosition
        client.simPause(False)
        return 1
        
    # Defines time step between each call for position data and plot
    def timeStep(self,totalTime = 0,client = airsim.CarClient()):
        idx = 0
        diff = 0
        data = stepData()
        #startTime = datetime.now().timestamp()
        startTime = client.getCarState().timestamp
        prevTime = startTime
        for step in np.arange(0,totalTime,self.timePeriod):
            car_state = client.getCarState()
            
            data.position.append((car_state.timestamp-prevTime)/1000000)
            self.Plot()
            if(step%(self.plotFreq/self.frequency)<=0.099):
                self.Plot()
                print(step)
            
            #print(data.position[idx])
            prevTime = car_state.timestamp
            idx = idx + 1
            currentTime = datetime.now().timestamp()
            diff = diff + currentTime - prevTime
            
            time.sleep(self.timePeriod)
        currentTime = datetime.now().timestamp()
        print(airsim.VehicleClient().getImuData())
        print(car_state)
        print("Duration: %f"%(car_state.timestamp - startTime))
            
    def main(self):
        # connect to the AirSim simulator
        client = airsim.CarClient()
        client.confirmConnection()
        client.enableApiControl(True)
        car_controls = airsim.CarControls()
        car_state = client.getCarState()
        VehicleClient = airsim.VehicleClient()
        sensor_state = VehicleClient.getImuData()
        #prevTime  = sensor_state.time_stamp/1000000000
        print("Set clockSpeed: %f" % self.clockSpeed)
        print("Frequency (After considering clockspeed): %f" %(self.frequency/self.clockSpeed))
        print("Speed %d, Gear %f \n" % (car_state.speed, car_state.gear))
        
        #time.sleep(2)
        #self.timeStep(2)
        sensor_state = VehicleClient.getImuData()
        currentTime  = sensor_state.time_stamp/1000000000 - 2
        prevTime = currentTime
        #print("Difference: %f"% (currentTime-prevTime))
       
        
        #Go forward + steer left
        car_controls.throttle = 0.5
        car_controls.steering = -0.5;
        client.setCarControls(car_controls)
        print("Turn left")
        car_state = client.getCarState()
        pos = car_state.kinematics_estimated.position
        self.startPosition = [pos.x_val,pos.y_val,pos.z_val]
        #VehicleClient.simPrintLogMessage("X position:",str(pos.x_val),1)
        #VehicleClient.simPrintLogMessage("Y position:",str(pos.y_val),1)
        #VehicleClient.simPrintLogMessage("Z position:",str(pos.z_val),1)
        #print("X position: %f   Y position: %f   Z position: %f"% (pos.x_val,pos.y_val,pos.z_val))
        
        
        #time.sleep(5)
        self.timeStep(5*self.clockSpeedMult)
        sensor_state = VehicleClient.getImuData()
        currentTime  = sensor_state.time_stamp/1000000000 - 5
        prevTime = currentTime
        #print("Difference: %f"% (currentTime-prevTime))
        
        #Go forward
        car_controls.throttle = 1
        car_controls.steering = 0
        client.setCarControls(car_controls)
        print("Go forward")
        prevTime = datetime.now().timestamp()
        self.timeStep(2.5*self.clockSpeedMult)
        #time.sleep(2.5)
        
        #self.Plot(client)
        
        sensor_state = VehicleClient.getImuData()
        currentTime = datetime.now().timestamp()
        # currentTime  = sensor_state.time_stamp/1000000000 - 10
        
        #print("Difference: %f"% (currentTime-prevTime))
        prevTime = currentTime
        # car_controls.brake = 0.3
        # client.setCarControls(car_controls)
        # time.sleep(3)
        # sensor_state = VehicleClient.getImuData()
        # currentTime  = sensor_state.time_stamp/1000000000 - 3
        # prevTime = currentTime
        # print("Difference: %f"% (currentTime-prevTime))
        
        #Go forward + steer right
        car_controls.throttle = 0.5
        car_controls.steering = 1;
        client.setCarControls(car_controls)
        print("Turn right")
        car_state = client.getCarState()
        pos = car_state.kinematics_estimated.position
        #print("X position: %f   Y position: %f   Z position: %f"% (pos.x_val,pos.y_val,pos.z_val))
        #time.sleep(3)
        self.timeStep(3*self.clockSpeedMult)
        
        #self.Plot(client)
        
        #Go forward
        car_controls.throttle = 0.5
        car_controls.steering = 0;
        client.setCarControls(car_controls)
        print("Go forward")
        car_state = client.getCarState()
        pos = car_state.kinematics_estimated.position
        #print("X position: %f   Y position: %f   Z position: %f"% (pos.x_val,pos.y_val,pos.z_val))
        #time.sleep(6)
        self.timeStep(6*self.clockSpeedMult)
        
        #self.Plot(client)
        
        #Go forward + steer right
        car_controls.throttle = 0.5
        car_controls.steering = 1;
        client.setCarControls(car_controls)
        print("Turn right")
        car_state = client.getCarState()
        pos = car_state.kinematics_estimated.position
        #print("X position: %f   Y position: %f   Z position: %f"% (pos.x_val,pos.y_val,pos.z_val))
        #time.sleep(3)
        self.timeStep(3*self.clockSpeedMult)
        
        #self.Plot(client)
        
        #Go forward
        car_controls.throttle = 0.5
        car_controls.steering = 0;
        client.setCarControls(car_controls)
        print("Go forward")
        car_state = client.getCarState()
        pos = car_state.kinematics_estimated.position
        #print("X position: %f   Y position: %f   Z position: %f"% (pos.x_val,pos.y_val,pos.z_val))
        #time.sleep(6)
        self.timeStep(6*self.clockSpeedMult)
        
        #self.Plot(client)
        
        car_controls.throttle = 0.5
        car_controls.steering = 1;
        client.setCarControls(car_controls)
        
        time.sleep(20)
        data = stepData()
        #print(data.position)
        client.reset()
        
        client.enableApiControl(False)
        return 1
ls = testingAK()
ls.main()    
      