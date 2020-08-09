# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:59:34 2020

@author: amk170930
"""

import numpy as np
import sys
import airsim
from airsim import Vector3r
import time

class mpcTest1AK:
    
    # The points by which the trajectory is defined
    trajPts = [Vector3r(0,-10,0),
                      Vector3r(0,-30,0),
                      Vector3r(20,-30,0),
                      Vector3r(40,-30,0),
                      Vector3r(70,-30,0),
                      Vector3r(70,-10,0),
                      Vector3r(70,0,0)]
    
    def plotLine(startPos, endPos, color = "red",thickness = 3, VehicleClient = airsim.VehicleClient()):

        rgba = [1.0, 0.0, 0.0, 1.0]
        if color == "blue":
            rgba = [0.0, 0.0, 1.0, 1.0]
            
        #Plot blue line from specified start point to end point
        VehicleClient.simPlotArrows(points_start = startPos,
                                    points_end = endPos,
                            color_rgba = rgba, arrow_size = 10, thickness = thickness, is_persistent = True)
    
    # Plots ellipse with given center
    # Assumption: Ellipses are only in X-Y plane ( Z is assumed constant)
    def plotEllipse(center,color = "red", a = 3,b = 1, nPoints = 20, VehicleClient = airsim.VehicleClient()):
        
        # a is semi major axis abd b is semi minor axis
        # nPoints is number of points you wish to define one half of ellipse with

        # Center
        h = center.x_val
        k = center.y_val
        
        # Coordinates at each step
        tailCoord = [0,0,center.z_val]
        headCoord = [0,0,center.z_val]
        
        #Create list of start and end points
        tailList = []
        headList = []
        
        # step is the difference in x value between each point plotted
        step = 2 * a / nPoints
        
        # Upper ellipse
        for i in range(nPoints):
            
            # Start position of the line
            tailCoord[0] = h - a + i * step
            tailCoord[1] = k + np.sqrt(b*b*(1-((tailCoord[0] - h)**2)/(a**2)))
            
            #End position of the line
            headCoord[0] = h - a + (i+1) * step
            headCoord[1] = k + np.sqrt(b*b*(1-((headCoord[0]-h)**2)/(a**2)))
            
            # Store the point
            tailList.append(Vector3r(tailCoord[0],tailCoord[1],tailCoord[2]))
            headList.append(Vector3r(headCoord[0],headCoord[1],headCoord[2]))
        
        # Lower ellipse
        for i in range(nPoints):
            # Start position of the line
            tailCoord[0] = h - a + i * step
            tailCoord[1] = k - np.sqrt(b*b*(1-((tailCoord[0] - h)**2)/(a**2)))
            
            # End position of the lineamsm
            headCoord[0] = h - a + (i+1) * step
            headCoord[1] = k - np.sqrt(b*b*(1-((headCoord[0] - h)**2)/(a**2)))
            
            # Store the point
            tailList.append(Vector3r(tailCoord[0],tailCoord[1],tailCoord[2]))
            headList.append(Vector3r(headCoord[0],headCoord[1],headCoord[2]))
        
        # Plot the ellipse
        mpcTest1AK.plotLine(tailList, headList, color)
            
    def main(self):
        
        #Setup airsim multirotor client
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        
        # Arm the drone
        print("arming the drone...")
        client.armDisarm(True)
        
        # Try taking off the multirotor to a deafult height of z = -3
        state = client.getMultirotorState()
        if state.landed_state == airsim.LandedState.Landed:
            print("taking off...")
            client.takeoffAsync().join()
        else:
            client.hoverAsync().join()

        time.sleep(1)

        state = client.getMultirotorState()
        if state.landed_state == airsim.LandedState.Landed:
            print("take off failed...")
            sys.exit(1)
        
        z = -5
        print("make sure we are hovering at %.1f meters..." %(-z))
        client.moveToZAsync(z, 1).join()
                
        # Locate current position of the multirotor
        pos = state.kinematics_estimated.position
        
        while pos.z_val >-4.99:
            state = client.getMultirotorState()
            pos = state.kinematics_estimated.position
        
        print("Height of %.2f meters reached"%(-z))
        # Initialize first point of trajectory with current position
        self.trajPts[0] = pos
        
        # Initialize all the z coordinates with current z coordinate
        for i in range(len(self.trajPts)):
            mpcTest1AK.trajPts[i].z_val = pos.z_val
        
        # List to store start points of the trajectory path
        trajStart = []
        # List to store end points of the trajectory path
        trajEnd = []
        
        # Loop to store the points in the required format for the trajectory
        for i in range(len(mpcTest1AK.trajPts)-1):
            
            # store the points 
            trajStart.append(mpcTest1AK.trajPts[i])
            trajEnd.append(mpcTest1AK.trajPts[i+1])
        
        # Plot the trajectory
        print("Plotting the trajectroy...")
        mpcTest1AK.plotLine(trajStart,trajEnd,"blue")
        
        time.sleep(2)
        
        # Plot the ellipses
        print("Plotting the ellipses...")
        for i in range(len(mpcTest1AK.trajPts)):
            mpcTest1AK.plotEllipse(mpcTest1AK.trajPts[i],"red")
        
        time.sleep(2)
        
        print("flying on path...")
        result = client.moveOnPathAsync([mpcTest1AK.trajPts[i] for i in range(len(mpcTest1AK.trajPts))],
                        4, 1200,
                        airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False,0), -1, 1).join()
        
        finalPt = mpcTest1AK.trajPts[len(mpcTest1AK.trajPts)-1]
        client.moveToPositionAsync(finalPt.x_val,finalPt.y_val,finalPt.z_val,1).join()
        time.sleep(10)
        
        print("landing...")
        client.landAsync().join()
        

        print("disarming...")
        client.armDisarm(False)
        
        client.enableApiControl(False)
        print("done.")

run = mpcTest1AK()
run.main()
        
            
            