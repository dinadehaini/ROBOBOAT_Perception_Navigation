# ROBOBOAT_Perception_Navigation


ROS Package Setup
Create a ROS Package: If you haven't created a ROS package yet, you can create one using: 

    mkdir -p ~/catkin_ws/src
    
    cd ~/catkin_ws/src 
  
    catkin_create_pkg buoy_detection rospy std_msgs sensor_msgs 


  Then, build your package and source the setup files:
  
    cd ~/catkin_ws
    
    catkin_make
    
    source devel/setup.bash
    
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
    
    source ~/.bashrc




  Prepare the Python Script:
  - Place your script in the scripts folder of your ROS package.
  - Make the script executable and build the package:

        chmod +x buoy_detection_node.py
        catkin_make


Test if the camera feed is accessible via ROS:

    rosrun image_view image_view image:=/camera/image_raw


Run Your Nodes:

    roslaunch your_package <launch_file>.launch


    
