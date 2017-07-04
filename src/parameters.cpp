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

#include "parameters.h"

int MAX_FEATURE_CNT= 200;
int NUM_OF_ITER = 10;
double CALIB_THRESHOLD_TIC = 0.0;
double CALIB_THRESHOLD_RIC = 1.0;
double INIT_DEPTH = 15;
double GRADIENT_THRESHOLD = 1.0;
double FEATURE_THRESHOLD = 5.0;
double MIN_PARALLAX = 10.0;
double MIN_PARALLAX_POINT =3.0 ;
double ERROR_THRESHOLD = 10000000000;
bool SHOW_HISTOGRAM = false ;
bool SHOW_GRAPH = false ;
bool SHOW_HTML = false;
bool MULTI_THREAD = false;
double IMU_RATE = 200 ;
double ACC_N = 0.01, ACC_W = 0.0002 ;
double GYR_N = 0.05, GYR_W = 4.0e-6;

std::vector<bool> RIC_OK, TIC_OK;
std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d GRAVITY{0.0, 0.0, 9.81007};

double BIAS_ACC_THRESHOLD = 0.5;
double BIAS_GYR_THRESHOLD = 0.1;
double SOLVER_TIME = 20;
bool COMPENSATE_ROTATION = true;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    MAX_FEATURE_CNT = readParam<int>(n, "max_feature_cnt");
    NUM_OF_ITER = readParam<int>(n, "num_of_iter");
    CALIB_THRESHOLD_RIC = readParam<double>(n, "calib_threshold_ric");
    CALIB_THRESHOLD_TIC = readParam<double>(n, "calib_threshold_tic");
    INIT_DEPTH = readParam<double>(n, "init_depth");
    GRADIENT_THRESHOLD = readParam<double>(n, "gradient_threshold") / FOCAL_LENGTH;
    FEATURE_THRESHOLD = readParam<double>(n, "feature_threshold") / FOCAL_LENGTH;
    MIN_PARALLAX = readParam<double>(n, "min_parallax") / FOCAL_LENGTH;
    MIN_PARALLAX_POINT = readParam<double>(n, "min_parallax_point") / FOCAL_LENGTH;
    ERROR_THRESHOLD = readParam<double>(n, "error_threshold");
    SHOW_HISTOGRAM = readParam<bool>(n, "show_histogram");
    SHOW_GRAPH = readParam<bool>(n, "show_graph");
    SHOW_HTML = readParam<bool>(n, "show_html");
    MULTI_THREAD = readParam<bool>(n, "multi_thread");

    IMU_RATE = readParam<double>(n, "imu_rate");
    ACC_N = readParam<double>(n, "acc_n");
    ACC_W = readParam<double>(n, "acc_w");
    GYR_N = readParam<double>(n, "gyr_n");
    GYR_W = readParam<double>(n, "gyr_w");
    BIAS_ACC_THRESHOLD = readParam<double>(n, "bias_acc_threshold");
    BIAS_GYR_THRESHOLD = readParam<double>(n, "bias_gyr_threshold");
    SOLVER_TIME = readParam<double>(n, "solver_time");
    COMPENSATE_ROTATION = readParam<bool>(n, "compensate_rotation");

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        RIC_OK.push_back(readParam<bool>(n, std::string("ric_ok") + std::to_string(i)));
        if (RIC_OK.back())
        {
            RIC.push_back(Utility::ypr2R(Eigen::Vector3d(
                readParam<double>(n, std::string("ric_y") + std::to_string(i)),
                readParam<double>(n, std::string("ric_p") + std::to_string(i)),
                readParam<double>(n, std::string("ric_r") + std::to_string(i)))));
            std::cout << RIC[i] << std::endl;
        }
        TIC_OK.push_back(readParam<bool>(n, std::string("tic_ok") + std::to_string(i)));
        if (TIC_OK.back())
        {
            TIC.push_back(Eigen::Vector3d(
                readParam<double>(n, std::string("tic_x") + std::to_string(i)),
                readParam<double>(n, std::string("tic_y") + std::to_string(i)),
                readParam<double>(n, std::string("tic_z") + std::to_string(i))));
        }
    }
}
