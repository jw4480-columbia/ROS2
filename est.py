#!/usr/bin/env python3

# Columbia Engineering
# MECS 4603 - Fall 2023

import math
import numpy
import time

import rclpy
from rclpy.node import Node

from state_estimator_msgs.msg import RobotPose
from state_estimator_msgs.msg import SensorData

class Estimator(Node):
    def __init__(self):
        super().__init__('estimator')

        # Publisher to publish state estimate
        self.pub_est = self.create_publisher(RobotPose, "/robot_pose_estimate", 1)

        # Initial estimates for the state and the covariance matrix
        self.x = numpy.zeros((3,1))
        self.P = numpy.zeros((3,3))

        # Covariance matrix for process (model) noise
        self.V = numpy.zeros((3,3))
        self.V[0,0] = 0.0025
        self.V[1,1] = 0.0025
        self.V[2,2] = 0.005

        self.step_size = 0.05

        # Subscribe to command input and sensory output of robot
        self.sensor_sub = self.create_subscription(SensorData, "/sensor_data", self.sensor_callback, 1)
        
    
    def estimate(self, sens: SensorData):
        '''This function gets called every time the robot publishes its control 
        input and sensory output. You must make use of what you know about 
        extended Kalman filters to come up with an estimate of the current
        state of the robot and covariance matrix. The SensorData message 
        contains fields 'vel_trans' and 'vel_ang' for the commanded 
        translational and rotational velocity respectively. Furthermore, 
        it contains a list 'readings' of the landmarks the robot can currently
        observe

        Args:
            sens: incoming sensor message
        '''
        #TODO: implement your extended Kalman filter here
        t=0.05
        x_pred=0
        y_pred=0
        th_pred=0

        v=sens.vel_trans
        w=sens.vel_ang

        #linearized F matrix
        F=numpy.zeros([3,3])
        F[0][0]=1
        F[0][2]=-1*v*t*numpy.sin(self.x[2])
        F[1][1]=1
        F[1][2]=v*t*numpy.cos(self.x[2])
        F[2][2]=1
        #F=numpy.array([[1,0,-1*v*t*numpy.sin(self.x[2])],[0,1,v*t*numpy.cos(self.x[2])],[0,0,1]])
        
        F_T=numpy.transpose(F)


        
        #Calculate system state space model
        x_pred=self.x[0]+v*t*numpy.cos(self.x[2])
        y_pred=self.x[1]+v*t*numpy.sin(self.x[2])
        th_pred=self.x[2]+w*t

        X_pred=numpy.zeros([3,1])
        X_pred[0]=x_pred
        X_pred[1]=y_pred
        X_pred[2]=th_pred
        
        P_pred=numpy.dot(numpy.dot(F,self.P),F_T)+self.V

        #sensors data
        readings=[]
        for i in range(len(sens.readings)):
            if sens.readings[i].range>=0.1:
                readings.append(sens.readings[i])


        N=len(readings)
        x_l=numpy.zeros([N,1])
        y_l=numpy.zeros([N,1])

        for i in range(N):
            x_l[i]=readings[i].landmark.x
            y_l[i]=readings[i].landmark.y

        H=numpy.zeros([2*N,3])
        y=numpy.zeros([2*N,1])
        W=numpy.identity(2*N)
        u=numpy.zeros([2*N,1])
        ##if there is any landmark
        if N>0:

            for i in range(N):
                W[2*i][2*i]=0.1
                W[2*i+1][2*i+1]=0.05
                H[2*i][0]=(X_pred[0]-x_l[i])/numpy.sqrt(numpy.power((X_pred[0]-x_l[i]),2)+numpy.power((X_pred[1]-y_l[i]),2))
                H[2*i][1]=(X_pred[1]-y_l[i])/numpy.sqrt(numpy.power((X_pred[0]-x_l[i]),2)+numpy.power((X_pred[1]-y_l[i]),2))
                H[2*i][2]=0
                H[2*i+1][0]=(y_l[i]-X_pred[1])/numpy.add(numpy.power((X_pred[0]-x_l[i]),2),numpy.power((X_pred[1]-y_l[i]),2))
                H[2*i+1][1]=(X_pred[0]-x_l[i])/numpy.add(numpy.power((X_pred[0]-x_l[i]),2),numpy.power((X_pred[1]-y_l[i]),2))
                H[2*i+1][2]=-1

                y[2*i]=readings[i].range
                y[2*i+1]=readings[i].bearing

                u[2*i]=y[2*i]-math.sqrt(numpy.power((X_pred[0]-x_l[i]),2)+numpy.power((X_pred[1]-y_l[i]),2))
                u[2*i+1]=y[2*i+1]-math.atan2(y_l[i]-X_pred[1],x_l[i]-X_pred[0])+X_pred[2]

                while u[2*i+1]>numpy.pi:
                    u[2*i+1]=u[2*i+1]-2*numpy.pi
                
                while u[2*i+1]<-1*numpy.pi:
                    u[2*i+1]=u[2*i+1]+2*numpy.pi
            
                H_T=numpy.transpose(H)
                S=numpy.add(numpy.dot(numpy.dot(H,P_pred),H_T),W)
                if N==1:
                    S_inv=numpy.linalg.pinv(S)
                else:
                    S_inv=numpy.linalg.inv(S)
                

                R=numpy.dot(numpy.dot(P_pred,H_T),S_inv)
        
                X_est=X_pred+numpy.dot(R,u)
                P_est=numpy.subtract(P_pred,numpy.dot(numpy.dot(R,H),P_pred))
                self.x[0]=X_est[0]
                self.x[1]=X_est[1]
                self.x[2]=X_est[2]
                self.P=P_est
        if N==0:
            self.x[0]=X_pred[0]
            self.x[1]=X_pred[1]
            self.x[2]=X_pred[2]
            self.P=P_pred

        
    
    def sensor_callback(self,sens):

        # Publish state estimate 
        self.estimate(sens)
        est_msg = RobotPose()
        est_msg.header.stamp = sens.header.stamp
        est_msg.pose.x = float(self.x[0])
        est_msg.pose.y = float(self.x[1])
        est_msg.pose.theta = float(self.x[2])
        self.pub_est.publish(est_msg)

def main(args=None):
    rclpy.init(args=args)   
    est = Estimator()
    rclpy.spin(est)
                
if __name__ == '__main__':
   main()

 
