# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 02:40:45 2020

@author: amk170930
"""


import airsim
import numpy as np
import setup_path 
import os
from datetime import datetime
import time

class frequencyTest:
# connect to the AirSim simulator 
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()
    
    
    start = time.time()
    prevTime = start

    car_state = client.getCarState()

    def carStateFreq(self):
        #Test variables
        revTime = 2 #seconds
        brakeTime = 1 #seconds
        tot = 0
        for idx in range(10):
    
            #Go reverse
            self.car_controls.throttle = -0.5
            self.car_controls.is_manual_gear = True;
            self.car_controls.manual_gear = -1
            self.car_controls.steering = 0
            self.client.setCarControls(self.car_controls)
            print("Go reverse")
            time.sleep(revTime)   # let car drive a bit
            self.car_controls.is_manual_gear = False; # change back gear to auto
            self.car_controls.manual_gear = 0  
        
            # apply brakes
            self.car_controls.brake = 1
            self.client.setCarControls(self.car_controls)
            print("Apply brakes")
            time.sleep(brakeTime)   # let car drive a bit
            self.car_controls.brake = 0 #remove brake
            
            #Time calculations
            currentTime = time.time()
            self.car_state = self.client.getCarState()
            diff = float((currentTime - self.prevTime - revTime - brakeTime)*1000)#miliseconds
            self.prevTime = currentTime
            
            freq = 1000/diff #Hertz
            tot = tot + freq
            print("Difference: %f Frequency: %f" % (diff,freq))
        print("\nAverage frequency: %f"% (tot/10.0))  
    
    def Freq():
        
        
        client = airsim.CarClient()
        VehicleClient = airsim.VehicleClient()
        sensor_state = VehicleClient.getImuData()
        car_controls = airsim.CarControls()
        testCases = 10
        revTime = 0#seconds
        
        time1 = time.time()
        for sensor in range(5):
            
            idx = 0
            tot = 0
            
            if sensor == 0:
                    print("\n\n\nIMU Data:")
            elif sensor ==1:
                    print("\n\n\nBarometer Data:")
            elif sensor == 2:
                    print("\n\n\nMagnetometer Data:")
            elif sensor == 3:
                    print("\n\n\nGps Data:")
            elif sensor == 4:
                    print("\n\n\nDistance Sensor Data:")
                    
            #prevTime  = datetime.now().timestamp()
            prevTime = sensor_state.time_stamp/1000000000
            while idx <=testCases:
                 #Go reverse
                car_controls.throttle = -0.5
                car_controls.is_manual_gear = True;
                car_controls.manual_gear = -1
                car_controls.steering = 0
                client.setCarControls(car_controls)
                #print("Go reverse")
                time.sleep(revTime)   # let car drive a bit
                car_controls.is_manual_gear = False; # change back gear to auto
                car_controls.manual_gear = 0  
                
                if sensor == 0:
                    sensor_state = VehicleClient.getImuData()
                elif sensor ==1:
                    sensor_state = VehicleClient.getBarometerData()
                elif sensor == 2:
                    sensor_state = VehicleClient.getMagnetometerData()
                elif sensor == 3:
                    sensor_state = VehicleClient.getGpsData()
                elif sensor == 4:
                    sensor_state = VehicleClient.getDistanceSensorData()
                    
               
                   
                
                #Time calculations
                #currentTime = datetime.now().timestamp()
                
                #car_state = client.getCarState()
                currentTime = sensor_state.time_stamp/1000000000 #convert nanoseconds to seconds
                diff = (((currentTime - prevTime)-revTime)*1000)#miliseconds
                prevTime = currentTime
                
                if diff !=0:
                    freq = 1000/diff #Hertz
                    tot = tot + freq
                else:
                    #print("0 difference encountered")
                    continue
                #print("Difference (In miliseconds): %f Frequency (Hz): %f" % (diff,freq))
                idx = idx + 1
            time2 = time.time()
            print("\nAverage frequency: %f"% (float(idx)/(time2-time1))) 
    

        
#frequencyTest.carStateFreq()
frequencyTest.Freq()
