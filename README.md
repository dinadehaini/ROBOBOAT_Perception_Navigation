# ROBOBOAT_Perception_Navigation


ROS Package Setup
Create a ROS Package: If you haven't created a ROS package yet, you can create one using: \n
 \t cd ~/catkin_ws/src \n
  \tcatkin_create_pkg buoy_detection rospy std_msgs sensor_msgs \n \n

  Then, build your package and source the setup files:
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash


  Prepare the Python Script:
    Place your script in the scripts folder of your ROS package.
    Make the script executable:
      chmod +x buoy_detection_node.py
