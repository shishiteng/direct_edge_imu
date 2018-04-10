/**
* This file is part of the implementation of our papers: 
* [1] Yonggen Ling, Manohar Kuse, and Shaojie Shen, "Direct Edge Alignment-Based Visual-Inertial Fusion for Tracking of Aggressive Motions", Autonomous Robots, 2017.
* [2] Yonggen Ling and Shaojie Shen, "Aggresive Quadrotor Flight Using Dense Visual-Inertial Fusion", in Proc. of the IEEE Intl. Conf. on Robot. and Autom., 2016.
* [3] Yonggen Ling and Shaojie Shen, "Dense Visual-Inertial Odometry for Tracking of Aggressive Motions", in Proc. of the IEEE International Conference on Robotics and Biomimetics 2015.
*
*
* For more information see <https://github.com/ygling2008/direct_edge_imu>
*
* This code is a free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This code is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this code. If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <fstream>
#include <boost/thread.hpp>
#include "src/parameters.h"
#include "src/LiveSLAMWrapper.h"
#include "src/settings.h"
#include "src/globalFuncs.h"
#include "src/SlamSystem.h"
#include "src/types.h"
#include "src/rosReconfigure.h"

#include <X11/Xlib.h>
#include "ros/ros.h"
#include "ros/package.h"
#include <glog/logging.h>

#include "visensor_node/visensor_imu.h"
#include "visensor_node/visensor_calibration.h"
#include "edge_imu_bias/ParamsConfig.h"
#include "sensor_msgs/Imu.h"

using namespace lsd_slam;
using namespace std ;

CALIBRATION_PAR calib_par ;
ros::Subscriber sub_image[2];
ros::Subscriber sub_imu;
LiveSLAMWrapper* globalLiveSLAM = NULL ;
cv::Mat R0, R1, P0, P1, Q;
cv::Rect roi1, roi2 ;
cv::Mat map00_, map01_, map10_, map11_ ;
int skipFrameNum = 0 ;
int cntFrame0Num = 0 ;
int cntFrame1Num = 0 ;

bool readEurocCalibration(string caliFilePath )
{
//cerr<<"11"<<endl;
    cv::FileStorage fs(caliFilePath.c_str(), cv::FileStorage::READ);
//cerr<<"12"<<endl;
    cv::FileNode cameraNode = fs["cameras"];
    bool gotCalibration ;
    if (cameraNode.isSeq() && cameraNode.size() > 0)
    {
//cerr<<"13"<<endl;
        int camIdx = 0;
        for (cv::FileNodeIterator it = cameraNode.begin();
             it != cameraNode.end(); ++it) {
//cerr<<"14"<<endl;
            if ((*it).isMap()
                    && (*it)["T_SC"].isSeq()
                    && (*it)["image_dimension"].isSeq()
                    && (*it)["image_dimension"].size() == 2
                    && (*it)["distortion_coefficients"].isSeq()
                    && (*it)["distortion_coefficients"].size() >= 4
                    && (*it)["distortion_type"].isString()
                    && (*it)["focal_length"].isSeq()
                    && (*it)["focal_length"].size() == 2
                    && (*it)["principal_point"].isSeq()
                    && (*it)["principal_point"].size() == 2) {
                ROS_INFO( "Found calibration in configuration file for camera %d", camIdx) ;
                gotCalibration = true;
            } else {
                return false;
            }
            ++camIdx;
        }
    }
    else{
        ROS_INFO("Did not find a calibration in the configuration file.") ;
    }

//cerr<<"15"<<endl;
    int image_width = 752;
    int image_height = 480;
    cv::Mat distortion_coefficients[2];
    double focal_length[2][2], principal_point[2][2] ;
    Eigen::Matrix4d T_c2i[2] ;
    if (gotCalibration)
    {
        int camIdx = 0;
        for (cv::FileNodeIterator it = cameraNode.begin();
             it != cameraNode.end(); ++it, camIdx++ )
        {

            ROS_INFO("read camera %d calibration parameteres", camIdx ) ;
//cerr<<"16"<<endl;
            cv::FileNode T_SC_node = (*it)["T_SC"];

            // extrinsics
            Eigen::Matrix4d T_SC;
            T_SC << T_SC_node[0], T_SC_node[1], T_SC_node[2], T_SC_node[3],
                    T_SC_node[4], T_SC_node[5], T_SC_node[6], T_SC_node[7],
                    T_SC_node[8], T_SC_node[9], T_SC_node[10], T_SC_node[11],
                    T_SC_node[12], T_SC_node[13], T_SC_node[14], T_SC_node[15];
            T_c2i[camIdx] = T_SC ;
            std::cout << T_c2i[camIdx] << "\n" ;

            cv::FileNode DC_node = (*it)["distortion_coefficients"] ;
            distortion_coefficients[camIdx] = cv::Mat(1, 4, CV_64F );
            for(int i = 0 ; i < 4 ; i++ ){
                distortion_coefficients[camIdx].at<double>(0, i) = DC_node[i] ;
            }
            std::cout << distortion_coefficients[camIdx] << "\n" ;

            cv::FileNode FL_node = (*it)["focal_length"] ;
            focal_length[camIdx][0] = FL_node[0] ;
            focal_length[camIdx][1] = FL_node[1] ;
            printf("%lf %lf\n", focal_length[camIdx][0], focal_length[camIdx][1] ) ;

            cv::FileNode PP_node = (*it)["principal_point"] ;
            principal_point[camIdx][0] = PP_node[0] ;
            principal_point[camIdx][1] = PP_node[1] ;
            printf("%lf %lf\n", principal_point[camIdx][0], principal_point[camIdx][1] ) ;
        }
    }

//cerr<<"17"<<endl;
    cv::Mat K0(3, 3, CV_64F ) ;
    cv::Mat K1(3, 3, CV_64F ) ;
    K0.setTo(0.0) ;
    K0.at<double>(0, 0) = focal_length[0][0] ;
    K0.at<double>(1, 1) = focal_length[0][1] ;
    K0.at<double>(0, 2) = principal_point[0][0] ;
    K0.at<double>(1, 2) = principal_point[0][1] ;
    K0.at<double>(2, 2) = 1.0 ;
    K1.setTo(0.0) ;
    K1.at<double>(0, 0) = focal_length[1][0] ;
    K1.at<double>(1, 1) = focal_length[1][1] ;
    K1.at<double>(0, 2) = principal_point[1][0] ;
    K1.at<double>(1, 2) = principal_point[1][1] ;
    K1.at<double>(2, 2) = 1.0 ;

    Eigen::Matrix4d T01 = T_c2i[1].inverse()*T_c2i[0] ;
    cv::Mat R01(3, 3, CV_64F) ;
    cv::Mat t01(3, 1, CV_64F) ;
    for( int i = 0 ;i < 3; i++ )
    {
        for( int j=0; j < 3; j++ )
        {
            R01.at<double>(i, j) = T01(i, j) ;
        }
        t01.at<double>(i, 0) = T01(i, 3) ;
    }
//cerr<<"18"<<endl;
    cv::Mat R0, P0, R1, P1, Q ;
    cv::Rect validRoi[2];
    cv::Size imageSize(image_width, image_height) ;
    cv::stereoRectify( K0, distortion_coefficients[0], K1, distortion_coefficients[1],
            imageSize, R01, t01, R0, R1, P0, P1, Q,
            cv::CALIB_ZERO_DISPARITY, 0, imageSize, &validRoi[0], &validRoi[1] ) ;

    cv::initUndistortRectifyMap(K0, distortion_coefficients[0], R0, P0, imageSize, CV_16SC2, map00_, map01_);
    cv::initUndistortRectifyMap(K1, distortion_coefficients[1], R1, P1, imageSize, CV_16SC2, map10_, map11_);
//cerr<<"19"<<endl;
    calib_par.fx[0] = P0.at<double>(0, 0)/2.0 ;
    calib_par.fy[0] = P0.at<double>(1, 1)/2.0 ;
    calib_par.cx[0] = (P0.at<double>(0, 2)+0.5)/2.0 - 0.5 ;
    calib_par.cy[0] = (P0.at<double>(1, 2)+0.5)/2.0 - 0.5 ;
//cerr<<"191"<<endl;
    for( int i = 0 ; i < 4; i++ ){
//cerr<<"191 d"<<i<<endl;
//cerr<<distortion_coefficients[0];
//cerr<<distortion_coefficients[1];
        calib_par.d[i] = distortion_coefficients[0].at<double>(0, i) ;
    }

    calib_par.width[0] = image_width/2 ;
    calib_par.height[0] = image_height/2 ;
    for( int i = 1 ; i < PYRAMID_LEVELS; i++ )
    {
//cerr<<"191 i:"<<i<<endl;
        calib_par.fx[i] = calib_par.fx[i-1]/2.0 ;
        calib_par.fy[i] = calib_par.fy[i-1]/2.0 ;
        calib_par.cx[i] = (calib_par.cx[i-1]+0.5)/2.0 - 0.5 ;
        calib_par.cy[i] = (calib_par.cy[i-1]+0.5)/2.0 - 0.5 ;
        calib_par.width[i] = calib_par.width[i-1]/2 ;
        calib_par.height[i] = calib_par.height[i-1]/2 ;
    }
//cerr<<"20"<<endl;
    Eigen::Matrix4d T_i2c = T_c2i[0].inverse() ;
    for( int i = 0 ; i < 3; i++ )
    {
        for( int j = 0 ; j < 3; j++ )
        {
            calib_par.R_i_2_c(i,j) = T_i2c(i, j) ;
        }
        calib_par.T_i_2_c(i) = T_i2c(i, 3) ;
    }
//cerr<<"21"<<endl;
    return true ;
}

void readCalibrationExtrinsics(string caliFilePath)
{
    cv::FileStorage fs(caliFilePath.c_str(), cv::FileStorage::READ);

    cv::Mat Ric_0, Tic_0;
    cv::Mat Ric_1, Tic_1;

    fs["Ric_0"] >> Ric_0 ;
    fs["Tic_0"] >> Tic_0 ;
    fs["Ric_1"] >> Ric_1 ;
    fs["Tic_1"] >> Tic_1 ;

    for( int i = 0 ; i < 3; i++ )
    {
        for( int j = 0 ; j < 3; j++ )
        {
            calib_par.R_i_2_c(i,j) = Ric_1.at<double>(i, j) ;
        }
        calib_par.T_i_2_c(i) = Tic_1.at<double>(0, i) ;
    }
    calib_par.R_i_2_c.setIdentity() ;
    cout << "extrinsics:" << endl ;
    cout << "R\n" << calib_par.R_i_2_c << endl ;
    cout << "T\n" << calib_par.T_i_2_c.transpose() << endl ;
}


void readCalibrationIntrisics(string caliFilePath)
{
    cv::FileStorage fs(caliFilePath.c_str(), cv::FileStorage::READ);

    cv::Mat D0, K0, D1, K1, R1, P1, R0, P0;

    fs["D0"] >> D0 ;
    fs["D1"] >> D1 ;
    fs["K0"] >> K0 ;
    fs["K1"] >> K1 ;
    fs["R0"] >> R0 ;
    fs["R1"] >> R1 ;
    fs["P0"] >> P0 ;
    fs["P1"] >> P1 ;

    int image_width = 752;
    int image_height = 480;
    cv::Size img_size(image_width, image_height);

    //cv::Mat K0_new = cv::getOptimalNewCameraMatrix(K0, D0, img_size, 0.0 ) ;
    //cv::Mat K1_new = cv::getOptimalNewCameraMatrix(K1, D1, img_size, 0.0 ) ;

    //cv::initUndistortRectifyMap
    cv::initUndistortRectifyMap(K0, D0, R0, P0, img_size, CV_16SC2, map00_, map01_);
    cv::initUndistortRectifyMap(K1, D1, R1, P1, img_size, CV_16SC2, map10_, map11_);
    /*
    calib_par.fx = K0.at<double>(0, 0)/2.0 ;
    calib_par.fy = K0.at<double>(1, 1)/2.0 ;
    calib_par.cx = (K0.at<double>(0, 2)+0.5)/2.0 - 0.5 ;
    calib_par.cy = (K0.at<double>(1, 2)+0.5)/2.0 - 0.5 ;
*/
    calib_par.fx[0] = P0.at<double>(0, 0)/2.0 ;
    calib_par.fy[0] = P0.at<double>(1, 1)/2.0 ;
    calib_par.cx[0] = (P0.at<double>(0, 2)+0.5)/2.0 - 0.5 ;
    calib_par.cy[0] = (P0.at<double>(1, 2)+0.5)/2.0 - 0.5 ;
    for( int i = 0 ; i < 4; i++ ){
        calib_par.d[i] = D0.at<double>(0, i) ;
    }

    calib_par.width[0] = image_width/2 ;
    calib_par.height[0] = image_height/2 ;


    for( int i = 1 ; i < PYRAMID_LEVELS; i++ )
    {
        calib_par.fx[i] = calib_par.fx[i-1]/2.0 ;
        calib_par.fy[i] = calib_par.fy[i-1]/2.0 ;
        calib_par.cx[i] = (calib_par.cx[i-1]+0.5)/2.0 - 0.5 ;
        calib_par.cy[i] = (calib_par.cy[i-1]+0.5)/2.0 - 0.5 ;
        calib_par.width[i] = calib_par.width[i-1]/2 ;
        calib_par.height[i] = calib_par.height[i-1]/2 ;
    }


    printf("fx=%f fy=%f cx=%f cy=%f\n", calib_par.fx[0], calib_par.fy[0], calib_par.cx[0], calib_par.cy[0] ) ;
    printf("height=%d width=%d\n", calib_par.width[0], calib_par.height[0] ) ;
}

//bool initCalibrationPar(string caliFilePath)
//{
//    //read calibration parameters
//    std::ifstream f(caliFilePath.c_str());
//    if (!f.good())
//    {
//        f.close();
//        printf(" %s not found!\n", caliFilePath.c_str());
//        return false;
//    }
//    std::string l1, l2;
//    std::getline(f,l1);
//    std::getline(f,l2);
//    f.close();

//    if(std::sscanf(l1.c_str(), "%f %f %f %f %f %f %f %f",
//                   &calib_par.fx, &calib_par.fy, &calib_par.cx, &calib_par.cy,
//                   &calib_par.d[0], &calib_par.d[1], &calib_par.d[2], &calib_par.d[3]) != 8 )
//    {
//        puts("calibration file format error 1") ;
//        return false ;
//    }
//    if(std::sscanf(l2.c_str(), "%d %d", &calib_par.width, &calib_par.height ) != 2)
//    {
//        puts("calibration file format error 2") ;
//        return false ;
//    }
//    printf("fx=%f fy=%f cx=%f cy=%f\n", calib_par.fx, calib_par.fy, calib_par.cx, calib_par.cy ) ;
//    printf("height=%d width=%d\n", calib_par.width, calib_par.height ) ;

//    return true ;
//}

void convertMsgToMatMono(const sensor_msgs::ImageConstPtr& msg, cv::Mat& img)
{
    int width = msg->width ;
    int height = msg->height ;
    img = cv::Mat(height, width, CV_8U);
    int k = 0 ;
    for( int i = 0 ; i < height ; i++ )
    {
        for ( int j = 0 ; j < width ; j++ )
        {
            img.at<uchar>(i, j) = msg->data[k] ;
            k++ ;
        }
    }
}

void image0CallBack(const sensor_msgs::ImageConstPtr& msg)
{
    if ( cntFrame0Num == 0 )
    {
        ros::Time tImage = msg->header.stamp;
        cv::Mat image ;
//cerr<<"image0CallBack:0"<<endl;
        convertMsgToMatMono(msg, image) ;
//cerr<<"image0CallBack:1"<<endl;
        cv::Mat imgRect ;

        //cout << "image0" << tImage << endl ;

        //double t = (double)cvGetTickCount()  ;
        //imgRect = image ;
        cv::remap(image, imgRect, map00_, map01_, cv::INTER_LINEAR);
        cv::pyrDown(imgRect, imgRect, cv::Size(imgRect.cols/2, imgRect.rows/2) ) ;

        //printf("rect time: %f\n", ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000) );

        //            cv::imshow("image0", imgRect ) ;
        //            cv::waitKey(1) ;

        globalLiveSLAM->image0_queue_mtx.lock();
        globalLiveSLAM->image0Buf.push_back(ImageMeasurement(tImage, imgRect));
        globalLiveSLAM->image0_queue_mtx.unlock();
    }
    cntFrame0Num++ ;
    if ( cntFrame0Num > skipFrameNum ){
        cntFrame0Num = 0 ;
    }
}

void image1CallBack(const sensor_msgs::ImageConstPtr& msg)
{   
    //    if ( cntFrame1Num == 0 )
    //    {
    //
    ros::Time tImage = msg->header.stamp;
    //cout << "image1 " << tImage << endl ;
    cv::Mat image ;
//cerr<<"image1CallBack:0"<<endl;
    convertMsgToMatMono(msg, image) ;
//cerr<<"image1CallBack:1"<<endl;
    cv::Mat imgRect ;

    //imgRect = image ;
    cv::remap(image, imgRect, map10_, map11_, cv::INTER_LINEAR);
    cv::pyrDown(imgRect, imgRect, cv::Size(imgRect.cols/2, imgRect.rows/2) ) ;

    //        cv::imshow("image1", imgRect ) ;
    //        cv::waitKey(1) ;

    globalLiveSLAM->image1_queue_mtx.lock();
    globalLiveSLAM->image1Buf.push_back(ImageMeasurement(tImage, imgRect));
    globalLiveSLAM->image1_queue_mtx.unlock();
    //    }
    //    cntFrame1Num++ ;
    //    if ( cntFrame1Num > skipFrameNum ){
    //        cntFrame1Num = 0 ;
    //    }
}

void imuCallback2(const sensor_msgs::ImuConstPtr& msg)
{
//cerr<<"imuCallback2:0"<<endl;
    visensor_node::visensor_imu imu_msg ;
    imu_msg.angular_velocity = msg->angular_velocity ;
    imu_msg.linear_acceleration.x = msg->linear_acceleration.x ;
    imu_msg.linear_acceleration.y = msg->linear_acceleration.y ;
    imu_msg.linear_acceleration.z = msg->linear_acceleration.z ;
    imu_msg.header = msg->header ;

    globalLiveSLAM->imu_queue_mtx.lock();
    globalLiveSLAM->imuQueue.push_back( imu_msg );
    globalLiveSLAM->imu_queue_mtx.unlock();
//cerr<<"imuCallback2:1"<<endl;
}

void imuCallBack(const visensor_node::visensor_imu& imu_msg )
{
//cerr<<"imuCallback:0"<<endl;
    globalLiveSLAM->imu_queue_mtx.lock();
    globalLiveSLAM->imuQueue.push_back( imu_msg );
    globalLiveSLAM->imu_queue_mtx.unlock();
//cerr<<"imuCallback:0"<<endl;
}

void process_image()
{
cerr<<"process_image:start"<<endl;
    globalLiveSLAM->Loop();
cerr<<"process_image:end"<<endl;
}

void process_BA()
{
cerr<<"process_BA:start"<<endl;
    globalLiveSLAM->BALoop();
cerr<<"process_BA:end"<<endl;
}

/*
void fun( ros::NodeHandle& nh)
{
    char path[1000] ;
    int image_height = 480 ;
    int image_width = 752 ;

    ros::Publisher pub_img0 = nh.advertise<sensor_msgs::Image>("/cam0/image_raw", 10);
    ros::Publisher pub_img1 = nh.advertise<sensor_msgs::Image>("/cam1/image_raw", 10);
    sensor_msgs::Image msg ;
    for( int i = 0 ; i < 60 ; i++ )
    {
        msg.header.stamp = ros::Time::now() ;

        sprintf(path, "/home/ygling2008/visensor_calibraion/calibrationdata/left-%04d.png", i ) ;
        cv::Mat img0 = cv::imread(path) ;
        cv::cvtColor(img0,img0,CV_BGR2GRAY);

        msg.header.frame_id = "cam0";
        msg.height = image_height ;
        msg.width = image_width ;
        sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::MONO8, image_height, image_width, image_width,
                                           img0.data );
        pub_img0.publish(msg) ;
        imshow("img0", img0) ;

        cv::waitKey(50) ;

        sprintf(path, "/home/ygling2008/visensor_calibraion/calibrationdata/right-%04d.png", i ) ;
        cv::Mat img1 = cv::imread(path) ;
        cv::cvtColor(img1,img1,CV_BGR2GRAY);
        msg.header.frame_id = "cam1";
        msg.height = image_height ;
        msg.width = image_width ;
        sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::MONO8, image_height, image_width, image_width,
                                           img1.data );
        printf("%s\n", path ) ;
        pub_img1.publish(msg) ;
        imshow("img1", img1) ;

        //usleep(100000) ;
        cv::waitKey(50) ;
    }
}
*/



int main( int argc, char** argv )
{
    XInitThreads();
    google::InitGoogleLogging(argv[0]);

    ros::init(argc, argv, "edge_imu_bias");
    ros::NodeHandle nh("~") ;

    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Fatal );

    readParameters(nh);

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_INFO("EIGEN_DONT_PARALLELIZE");
#endif

    //cerr<<"1"<<endl;

    dynamic_reconfigure::Server<edge_imu_bias::ParamsConfig> srv(ros::NodeHandle("~"));
    srv.setCallback(dynConfCb);

    string intrinsics_calibration_file ;
    string extrinsics_calibration_file ;
    string euroc_calibration_file ;

    //string packagePath = ros::package::getPath("edge_imu_bias")+"/";
    string packagePath = "/home/sst/catkin_ws_master/src/direct_edge_imu/" ;
    nh.param("intrinsics_calibration_file", intrinsics_calibration_file, packagePath+"/calib/combine2_icra.yml" ) ;
    nh.param("extrinsics_calibration_file", extrinsics_calibration_file, packagePath+"/calib/visensor.yml" ) ;
    nh.param("euroc_calibration_file", euroc_calibration_file, packagePath+"/calib/config_euroc.yaml" ) ;
    nh.param("skipFrameNum", skipFrameNum, 0 ) ;
    nh.param("onUAV", onUAV, false ) ;
    nh.param("enable_histogramEqualization", enable_histogramEqualization, false ) ;
    nh.param("errorTranslationThreshold", errorTranslationThreshold, 1.0 ) ;
    nh.param("errorAngleThreshold", errorAngleThreshold, 10.0/180*PI ) ;
    nh.param("adaptiveCannyThreshold", adaptiveCannyThreshold, false ) ;
    nh.param("printDebugInfo", printDebugInfo, true ) ;
    nh.param("enable_LoopClosure", enable_LoopClosure, true ) ;
    nh.param("frontMarginalization", frontMarginalization, false ) ;
    nh.param("loopClosureInterval", loopClosureInterval, 0.1 ) ;
    nh.param("visualWeight", visualWeight, 10000.0 ) ;

    nh.param("bias_g_x_0", bias_g_x_0, 0.0 ) ;
    nh.param("bias_g_y_0", bias_g_y_0, 0.0 ) ;
    nh.param("bias_g_z_0", bias_g_z_0, 0.0 ) ;
    nh.param("bias_a_x_0", bias_a_x_0, 0.0 ) ;
    nh.param("bias_a_y_0", bias_a_y_0, 0.0 ) ;
    nh.param("bias_a_z_0", bias_a_z_0, 0.0 ) ;
    nh.param("denseOrNot", denseOrNot, true ) ;
    nh.param("edgeIterationNum", edgeIterationNum, 10 ) ;
    nh.param("huber_r_v", huber_r_v, 0.05 ) ;
    nh.param("huber_r_w", huber_r_w, 1.0 ) ;
    nh.param("IMUorNot", IMUorNot, true ) ;
    huber_r_w = huber_r_w*PI/180 ;
    if (IMUorNot == false ){
        huber_r_v *= 3.0 ;
        huber_r_w *= 3.0 ;
    }

//cerr<<"2"<<endl;

    readEurocCalibration(euroc_calibration_file) ;
    //readCalibrationIntrisics( intrinsics_calibration_file ) ;
    //readCalibrationExtrinsics( extrinsics_calibration_file );
//cerr<<"3"<<endl;
    cntFrame0Num = cntFrame1Num = 0 ;
    //sub_image[0] = nh.subscribe("/cam1/image_raw", 100, &image0CallBack );
    //sub_image[1] = nh.subscribe("/cam0/image_raw", 100, &image1CallBack );
    //sub_imu = nh.subscribe("/cust_imu0", 1000, &imuCallBack ) ;
    sub_image[0] = nh.subscribe("/cam0/image_raw", 100, &image0CallBack );
    sub_image[1] = nh.subscribe("/cam1/image_raw", 100, &image1CallBack );
    sub_imu = nh.subscribe("/imu0", 1000, &imuCallback2 ) ;

//cerr<<"4"<<endl;
    LiveSLAMWrapper slamNode(packagePath, nh, &calib_par );
    globalLiveSLAM = &slamNode ;
    globalLiveSLAM->popAndSetGravity( skipFrameNum );
    boost::thread ptrProcessImageThread = boost::thread(&process_image);
    boost::thread ptrProcessBAThread = boost::thread(&process_BA);

//cerr<<"4"<<endl;

    ros::spin() ;
    ptrProcessImageThread.join();
    ptrProcessBAThread.join();

    return 0;
}
