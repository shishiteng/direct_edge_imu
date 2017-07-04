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


#pragma once
#include <vector>
#include <mutex>
#include "settings.h"
#include "rosPub.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
//#include "marginalization.h"
#include "utility.h"
#include "globalFuncs.h"
#include "types.h"
#include "myMath.h"
#include "SolveDVO.h"

#include "parameters.h"
#include "pose_local_parameterization.h"
#include "imu_factor.h"
#include "marginalization_factor.h"
#include "relative_pose_factor.h"
#include "Tracking/TrackingReference.h"
#include "Tracking/SE3Tracker.h"
#include "edge_alignment_factor.h"

using namespace std ;

class SlamSystem
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //typedef EulerIntegration Integration_t;
    typedef MidpointIntegration Integration_t;
    //typedef RK4Integration Integration_t;
    typedef IMUFactor<Integration_t> IMUFactor_t;

    // settings. Constant from construction onward.
    int width;
    int height;
    Eigen::Matrix3f K;
    //SE3 RefToFrame ;
    Eigen::Vector3d gravity;

    //MARGINALIZATION margin;
    CALIBRATION_PAR* calib_par ;
    Eigen::Matrix3d R_i_2_c;
    Eigen::Vector3d T_i_2_c;

    bool trackingIsGood;
    Math math;


    // used in nonlinear, fixed first state
    const double prior_p_std = 0.0001;
    const double prior_q_std = 0.01 / 180.0 * M_PI;

    // used in nonlinear strong assumption
    const double tic_std = 0.005;
    const double ric_std = 0.01 / 180.0 * M_PI;

    // from imu_3dm_gx4
    const double acc_density = 1.0e-3;
    const double gyr_density = 8.73e-5;
    const double update_rate = 200.0;

    const Matrix3d acc_cov = std::pow(acc_density * std::sqrt(update_rate), 2.0) * Matrix3d::Identity(); // 0.014
    const Matrix3d gyr_cov = std::pow(gyr_density * std::sqrt(update_rate), 2.0) * Matrix3d::Identity(); // 0.0012
    const Matrix2d pts_cov = (0.5 / FOCAL_LENGTH) * (0.5 / FOCAL_LENGTH) * Matrix2d::Identity();
    const Matrix3d gra_cov = 0.001 * 0.001 * Matrix3d::Identity();

    int frame_count ;
    Vector3d acc_0, gyr_0;
    Vector3d Ps[WINDOW_SIZE + 1];
    Vector3d Vs[WINDOW_SIZE + 1];
    Matrix3d Rs[WINDOW_SIZE + 1];
    Vector3d Bas[WINDOW_SIZE + 1];
    Vector3d Bgs[WINDOW_SIZE + 1];
    STATE* states[WINDOW_SIZE + 1] ;
    STATE* pKF = nullptr;
    STATE* lastKF = nullptr ;
    int nTrackFrame ;
    STATE currentFrame ;
    std::mutex KF_mutex ;

    bool twoWayMarginalizatonFlag = false;//false, marginalize oldest; true, marginalize newest
    bool marginalization_flag;

    //ceres
    int ceres_frame_count;
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    IMUFactor_t *imu_factors[WINDOW_SIZE + 1];
    MarginalizationFactor *last_marginalization_factor;
    vector<double *> last_marginalization_parameter_blocks;

    cv::Ptr<cv::StereoBM> bm_ ;
    cv::Mat gradientMapForDebug ;

    int frameInfoListHead, frameInfoListTail ;
    FRAMEINFO frameInfoList[frameInfoListSize] ;
    std::mutex frameInfoList_mtx;

    bool lock_densetracking = false ;
    std::mutex tracking_mtx;

    bool onTracking = false ;
    std::mutex onTracking_mtx ;

    SlamSystem(int w, int h, Eigen::Matrix3f K, CALIBRATION_PAR* cali);
    ~SlamSystem();

    // tracks a frame.
    // first frame will return Identity = camToWord.
    // returns camToWord transformation of the tracked frame.
    // frameID needs to be monotonically increasing.
    void trackFrameEdge(cv::Mat image0, cv::Mat image1, unsigned int frameID,
                    ros::Time imageTimeStamp, Matrix3d deltaR);

    void trackFrameDense(cv::Mat image0, unsigned int frameID,
                    ros::Time imageTimeStamp, Matrix3d deltaR);

    void initGravity( const Eigen::Vector3d& gravity_b0 );
    /** Returns the current pose estimate. */
    void debugDisplayDepthMap();
    void insertFrame(ros::Time time, cv::Mat img) ;
    void insertCameraLink(STATE *keyFrame, STATE *currentFrame,
                          const Matrix3d& R_k_2_c, const Vector3d& T_k_2_c,
                          const Eigen::Matrix<double, 6, 6>& lastestATA );
    void processIMU(double dt, const Vector3d&linear_acceleration, const Vector3d &angular_velocity);
    void setReprojectionListRelateToLastestKeyFrameEdge();
    void setReprojectionListRelateToLastestKeyFrameEdgeCeres();
    void optimizeWithEdgeAlignment(STATE *keyFrame, STATE *currentFrame,
                                   Matrix3d& R_i_2_j, Vector3d& t_i_2_j,
                                   CALIBRATION_PAR *cali, int level, double &ratio, double &aveError);
    void setReprojectionListRelateToLastestKeyFrameDense();
    void para2eigen() ;
    void eigen2para() ;
    void solve_ceres() ;
    void moveData() ;
    void marginalization();
    void generateDubugMap(STATE* currentFrame, cv::Mat& gradientMapForDebug ) ;

    SolveDVO dvo;
    SolveDVO dvoLoop ;
    Eigen::VectorXf epsilonVec;
    Eigen::MatrixXf reprojections;
    Eigen::VectorXf energyAtEachIteration;
    Eigen::Matrix3d cR_64 ;
    Eigen::Vector3d cT_64 ;

    int bestEnergyIndex = -1;
    double visibleRatio = 0.0;

    //std::shared_ptr<Frame> lastTrackedFrame;
    bool createNewKeyFrame;


    //for dense tracking
    SE3 RefToFrame ;
    std::shared_ptr<lsd_slam::Frame> currentKeyFrame ;
    float msTrackFrame, msOptimizationIteration, msFindConstraintsItaration, msFindReferences;
    int nOptimizationIteration, nFindConstraintsItaration, nFindReferences;
    float nAvgTrackFrame, nAvgOptimizationIteration, nAvgFindConstraintsItaration, nAvgFindReferences;
    struct timeval lastHzUpdate;
    lsd_slam::TrackingReference* trackingReference; // tracking reference for current keyframe. only used by tracking.
    lsd_slam::SE3Tracker* tracker;
    lsd_slam::TrackingReference* trackingReferenceConstraint; // tracking reference for current keyframe. only used by tracking.
    lsd_slam::SE3Tracker* trackerConstraint;
    float tracking_lastResidual;
    float tracking_lastUsage;
    float tracking_lastGoodPerBad;
    float tracking_lastGoodPerTotal;
    int lastNumConstraintsAddedOnFullRetrack;
    float lastTrackingClosenessScore;
    void updateTrackingReference();
};
