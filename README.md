This is the webpage of our work about visual-inertial fusion. It consists in part of the implementations in:

[1] Yonggen Ling, Manohar Kuse, and Shaojie Shen, "Direct Edge Alignment-Based Visual-Inertial Fusion for Tracking of Aggressive Motions", Autonomous Robots, 2017.
[2] Yonggen Ling, and Shaojie Shen, "Aggresive Quadrotor Flight Using Dense Visual-Inertial Fusion", in Proc. of the IEEE Intl. Conf. on Robot. and Autom., 2016.
[3] Yonggen Ling, and Shaojie Shen, "Dense Visual-Inertial Odometry for Tracking of Aggressive Motions", in Proc. of the IEEE International Conference on Robotics and Biomimetics 2015.

The source code is released under GPLv3 license.

For the first work [1], this video shows how it work: https://onedrive.live.com/redir?resid=907AC500FCC6D19C!2904&authkey=!ANm_wASUQNGdhY8&ithint=video%2cmp4 (720p HD mode is recommanded.)

For the second and third work [2][3], please visit https://github.com/ygling2008/dense_new for how it works. The implementations here is an extension of [2][3], such as jointly estimating imu biases and robot poses, adding the flexibity to open/close front marginalization. More details can be found in the launch files and codes.

Our package is compatible with the standard driver of VI-Sensor and ROS version of indigo. OpenCV is needed. ROS packages related to VI-sensor are needed, too. (https://github.com/ethz-asl/libvisensor and https://github.com/ethz-asl/visensor_node.git).

After downloading the codes, use catkin_make to complier it. Then type "roslaunch edge_imu_bias V1_01_easy.launch 
" to run the Euroc Dataset(http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets). You need to adjust the path of the downloaded Euroc dataset in your computer. 

NOTE: The flag (denseOrNot) in the launch file determines whether to use direct dense tracking (true) or edge alignment (false). 

For more questions, please contact ylingaa at connect dot ust dot hk .
