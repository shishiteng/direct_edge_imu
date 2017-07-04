#pragma once

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>
#include "image_transport/image_transport.h"
#include "sensor_msgs/image_encodings.h"
#include "sensor_msgs/PointCloud.h"
#include "boost/thread.hpp"
#include "types.h"
#include "rosPub.h"
#include "visensor_node/visensor_imu.h"
#include <tf/transform_broadcaster.h>
#include "SlamSystem.h"
#include "SophusUtil.h"

struct LiveSLAMWrapper
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    LiveSLAMWrapper(std::string packagePath, ros::NodeHandle& _nh, CALIBRATION_PAR *calib_par);

	/** Destructor. */
	~LiveSLAMWrapper();
	
    void popAndSetGravity(int skipFrameNum) ;
    void initRosPub();

	/** Runs the main processing loop. Will never return. */
	void Loop();

    /** Runs the main processing loop. Will never return. */
    void BALoop();

    void pubCameraLink() ;

    //void pubPointCloud(int num, ros::Time imageTimeStamp , Matrix3d R_vi_2_odometry) ;
	
    std::list<ImageMeasurement> image0Buf;
    std::list<ImageMeasurement> image1Buf;
    std::list<ImageMeasurement>::iterator pImage0Iter;
    std::list<ImageMeasurement>::iterator pImage1Iter;
    boost::mutex image0_queue_mtx;
    boost::mutex image1_queue_mtx;

    std::list<visensor_node::visensor_imu> imuQueue;
    std::list<visensor_node::visensor_imu>::iterator currentIMU_iter;
    boost::mutex imu_queue_mtx;

	// initialization stuff
	bool isInitialized;
    Eigen::Vector3d gravity_b0 ;
    Eigen::Matrix3d R_vi_2_odometry ;
    Eigen::Matrix3d R_i_2_c ;
    Eigen::Vector3d T_i_2_c ;

    // Odometry
    SlamSystem* Odometry;
    CALIBRATION_PAR *cali ;

	std::string outFileName;
    std::ofstream outFile;
	
	float fx, fy, cx, cy;
    double sumDist ;
	int width, height;

	int imageSeqNumber;

    ros::Time lastLoopClorsureTime ;
    ros::Time initialTime ;

    int cnt_info_smooth ;
    geometry_msgs::Vector3 to_pub_info ;

    ros::NodeHandle nh ;
    ros::Publisher pub_path ;
    ros::Publisher pub_odometry ;
    ros::Publisher pub_pose ;
    ros::Publisher pub_cloud ;
    ros::Publisher pub_grayImage ;
    ros::Publisher pub_resudualMap ;
    ros::Publisher pub_edge ;
    ros::Publisher pub_distanceTransform ;
    ros::Publisher pub_gradientMapForDebug ;
    ros::Publisher pub_reprojectMap ;
    ros::Publisher pub_denseTracking ;
    ros::Publisher pub_angular_velocity ;
    ros::Publisher pub_linear_velocity ;
    ros::Publisher pub_nav_Odometry ;
    visualization_msgs::Marker path_line;
};
