"""Importing float32, float64, string, and int16 data types to initialize variables with
Importing ROS, time, numpy, math
Importing IMU"""


#!/usr/bin/env python
import rclpy
from rclpy.node import Node
import time
import numpy as np
from std_msgs.msg import Float32
from std_msgs.msg import Float64
from std_msgs.msg import String
from std_msgs.msg import Int16
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Imu
import timer
# from geometry_msgs.msg import Quaternion
# from geometry_msgs.msg import Pose
# from geometry_msgs.msg import Point
# from geographic_msgs.msg import WayPoint
# from geographic_msgs.msg import GeoPoint
# from geographic_msgs.msg import KeyValue
# from geographic_msgs.msg import GeoPath
# from geographic_msgs.msg import GeoPoseStamped


import math

"""Initialize variables/constants:
Current latitude/longitude (deg)
Target angle/lat/longitude (deg)
Booleans for arrival status, waypoint start and finish
Target distance (m)
Minimum distance constant (threshold for meeting waypoint)
Angle threshold of 10 degrees
"""


class auto_nav(Node):

    def __init__(self):
        super().__init__('navigator')
        # self.current_lat = 0
        # self.current_lon = 0
        self.target_angle = 0.0
        # self.target_lon = 0
        # self.target_lat = 0
        # self.arrived = True
        # self.waypoint_done = False
        # self.waypoint_started = False
        # self.station_started = False
        self.target_distance = 0
        # TODO: tune threshold
        self.MIN_DIST = 0.00003 #in lat/lon
        # TODO: finetune
        self.ANGLE_THR = 10 # angle in degrees
        self.arrived_pub = self.create_publisher(String, '/wamv/navigation/arrived', 10)
        self.mc_torqeedo = self.create_publisher(String, '/wamv/torqeedo/motor_cmd', 10)
        self.navigation_input=self.create_publisher(String,'navigation_input',10)
        self.waypoint_list = []
        self.waypoint_index = 0
        print("Navigator Init done")


    def test_move(self):
        print("TESTING THRUSTERS")
        while time.sleep(5):
            dir_to_move = "w"
            #publish go straight command
            msg.data=dir_to_move
            self.navigation_input.publish(msg)
            print("go straight")
            self.initial_alignment = False
        
        print("arrived / stop")
        dir_to_move = "s"
        msg.data=dir_to_move
        self.navigation_input.publish(msg)
        

    def main(args=None):
        rclpy.init()
        navigator = auto_nav()
        navigator.test_move()
        rclpy.spin(navigator)

    if __name__ == '__main__':
            main()







        # msg=String()
        # if self.arrived:
        #     self.waypoint_done = True
        #     #publish the stop command
        #     print("Arrived , not going further")
        #     dir_to_move = "s"
        #     msg.data=dir_to_move
        #     self.navigation_input.publish(msg)
        #     return

        # # TODO: PID pass through / target
        # if ((angle_diff) > angle_thr):
        #     dir_to_move = "d"
        #     #publish CW command
        #     msg.data=dir_to_move
        #     self.navigation_input.publish(msg)
        #     print("turning clockwise")

        #     #Added Coach Amit's delay code
        #     time.sleep(0.5)
        #     dir_to_move = "w"
        #     #publish forward command
        #     msg.data=dir_to_move
        #     self.navigation_input.publish(msg)
        #     print("continue after turning clockwise")
            
        # elif ((angle_diff) < -angle_thr):
        #     dir_to_move = "a"
        #     #publish CCW/ACW command
        #     msg.data=dir_to_move
        #     self.navigation_input.publish(msg)
        #     print("turning anticlockwise")
            
        #     #Added Coach Amit's delay code
        #     time.sleep(0.5)
        #     dir_to_move = "w" #publish forward command
        #     msg.data=dir_to_move
        #     self.navigation_input.publish(msg)
        #     print("continue after turning anti-clockwise")
        # else:
        #     if(not self.arrived):
        #         dir_to_move = "w"
        #         #publish go straight command
        #         msg.data=dir_to_move
        #         self.navigation_input.publish(msg)
        #         print("go straight")
        #         self.initial_alignment = False
        #     else: #stop the boat
        #         print("arrived / stop")
        #         dir_to_move = "s"
        #         msg.data=dir_to_move
        #         self.navigation_input.publish(msg)
