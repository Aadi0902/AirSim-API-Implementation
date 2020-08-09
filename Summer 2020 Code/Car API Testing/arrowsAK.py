# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 01:39:24 2020

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

class arrowsAK:
    startPosition = [0,0,0]
    endPosition = [-0.5,0.5,0]
    
    clockSpeed = 1 #From settings.json
    clockSpeedMult = 1/clockSpeed
    
    frequency = 5
    timePeriod = 1/(frequency*clockSpeed)
    
    #Plots red arrows from start point to a given point
    def Plot(self,client = airsim.CarClient()):
        #client.simPause(True)
        car_state = client.getCarState()
        VehicleClient = airsim.VehicleClient()
        pos = car_state.kinematics_estimated.position
        
        #Plot red arrows persistently
        VehicleClient.simPlotArrows(points_start = [Vector3r(self.startPosition[0],self.startPosition[1],self.startPosition[2])],
                                    points_end = [Vector3r(self.endPosition[0],self.endPosition[1],self.endPosition[2])],
                            color_rgba = [1.0, 0.0, 0.0, 1.0], arrow_size = 10, thickness = 5, is_persistent = True)
        self.startPosition = self.endPosition
        self.endPosition[0]-= 0.05
        self.endPosition[1]+= 0.05
        #client.simPause(False)
        return 1
    def main(self):
        for i in range(100):
            self.Plot()
            time.sleep(self.timePeriod)
        print("DONE")
            
ls = arrowsAK()
ls.main()
        