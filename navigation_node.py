#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from your_custom_msg.msg import BuoyDetection

def detection_callback(msg):
    command = Twist()
    # Simple logic to navigate based on detection (obviously replace w real logic)
    command.linear.x = 1.0  # speed value
    command.angular.z = 0.5  # steering value (look into vector stuff they were talking abt?)
    pub.publish(command)

rospy.init_node('navigation_node')
rospy.Subscriber('/buoy_detections', BuoyDetection, detection_callback)
pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
rospy.spin()
