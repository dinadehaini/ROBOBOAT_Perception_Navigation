#!/usr/bin/env python
import rospy
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from your_custom_msg.msg import BuoyDetection  # Need a custom message for detections define this message in ROS package

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov8', 'custom', path='path_to_your_model.pt') # add path to model
model.eval()

# Initialize CV Bridge
bridge = CvBridge()

def image_callback(msg):
    rospy.loginfo("Received an image!")
    try:
        # Convert ROS image message to CV2 format
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)

    # Inference
    results = model(cv_image)
    detections = results.pandas().xyxy[0]  # Process detection results

    # Assuming BuoyDetection message has fields like x_min, y_min, x_max, y_max
    detection_msg = BuoyDetection()
    for index, row in detections.iterrows():
        detection_msg.x_min = row['xmin']
        detection_msg.y_min = row['ymin']
        detection_msg.x_max = row['xmax']
        detection_msg.y_max = row['ymax']
        pub.publish(detection_msg)

def main():
    rospy.init_node('buoy_detection_node', anonymous=True)
    rospy.Subscriber("/camera/image_raw", Image, image_callback)
    global pub
    pub = rospy.Publisher('/buoy_detections', BuoyDetection, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    main()
