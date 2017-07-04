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

#include "SlamSystem.h"
#include "settings.h"
#include "LiveSLAMWrapper.h"
#include "sensor_msgs/PointCloud2.h"
#include "quadrotor_msgs/Odometry.h"

// for mkdir
#include <sys/types.h>
#include <sys/stat.h>
#include "opencv2/opencv.hpp"

using namespace Eigen;

SlamSystem::SlamSystem(int w, int h, Eigen::Matrix3f K, CALIBRATION_PAR* cali )
{
    //    if(w%16 != 0 || h%16!=0)
    //    {
    //        printf("image dimensions must be multiples of 16! Please crop your images / video accordingly.\n");
    //        assert(false);
    //    }
    calib_par = cali ;
    this->width = w;
    this->height = h;
    this->K = K;
    trackingIsGood = true;
    lock_densetracking = false ;

    createNewKeyFrame = false;
    onTracking = false ;

    dvo.K = K;
    dvo.K_inv = K.inverse() ;

    dvo.fx = K(0, 0) ;
    dvo.fy = K(1, 1) ;
    dvo.cx = K(0, 2) ;
    dvo.cy = K(1, 2) ;

    int maxDisparity = 64 ;
    int blockSize = 13 ;
    bm_ = cv::StereoBM::create( maxDisparity, blockSize );

    frameInfoListHead = frameInfoListTail = 0 ;

    //clear state
    ceres_frame_count = -1 ;
    frame_count = 0 ;
    last_marginalization_factor = nullptr ;
    for (int i = 0; i <= WINDOW_SIZE ; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        imu_factors[i] = nullptr;
        states[i] = new STATE() ;
    }

    //init ceres
    loss_function = new ceres::CauchyLoss(1.0);
    //    for (int i = 0; i <= ceres_frame_count; i++)
    //    {
    //        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    //        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
    //        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    //    }
    pKF = states[0] ;
    pKF->state_id = 0 ;

#ifdef DENSE
    currentKeyFrame = nullptr ;
    tracker = new lsd_slam::SE3Tracker(w,h,K);
    trackingReference = new lsd_slam::TrackingReference();

    trackerConstraint = new lsd_slam::SE3Tracker(w,h,K);
    trackingReferenceConstraint = new lsd_slam::TrackingReference();
    trackingIsGood = true;
#endif
}

SlamSystem::~SlamSystem()
{
}

void SlamSystem::generateDubugMap(STATE* currentFrame, cv::Mat& gradientMapForDebug )
{
    int n = currentFrame->frame->height() ;
    int m = currentFrame->frame->width() ;
    const float* pIdepth = currentFrame->frame->idepth(0) ;
    for ( int i = 0 ; i < n ; i++ )
    {
        for( int j = 0 ; j < m ; j++ )
        {
            if (  *pIdepth > 0 ){
                gradientMapForDebug.at<cv::Vec3b>(i, j)[0] = 0;
                gradientMapForDebug.at<cv::Vec3b>(i, j)[1] = 255;
                gradientMapForDebug.at<cv::Vec3b>(i, j)[2] = 0;
            }
            pIdepth++ ;
        }
    }
}

void SlamSystem::initGravity( const Eigen::Vector3d& gravity_b0 )
{
    ROS_INFO_STREAM("gravity init " << gravity_b0.transpose());
    Rs[0] = Utility::g2R(gravity_b0);
    ROS_ASSERT( Utility::R2ypr(Rs[0]).x() <= 1e-6 );
    Vector3d gravity_new = Rs[0] * gravity_b0.normalized() * GRAVITY.norm();
    ROS_INFO_STREAM("reset gravity " << gravity_new.transpose());
    ROS_INFO_STREAM("origin gravity " << (Rs[0].transpose() * gravity_new).transpose());
    ROS_INFO_STREAM("R0:\n" << Rs[0]);

    Bgs[0] << bias_g_x_0, bias_g_y_0, bias_g_z_0 ;
    Bas[0] << bias_a_x_0, bias_a_y_0, bias_a_z_0 ;
    //    Bgs[0] << -0.002295, 0.024939, 0.081667;
    //    Bas[0] << -0.023601, 0.121044, 0.074783;



//    ROS_INFO_STREAM("gravity init " << g.transpose());
//    Rs[0] = Utility::g2R(g);
//    ROS_ASSERT(Utility::R2ypr(Rs[0]).x() <= 1e-6);
//    g = Rs[0] * g.normalized() * G.norm();
//    ROS_INFO_STREAM("reset gravity " << g.transpose());
//    ROS_INFO_STREAM("origin gravity " << (Rs[0].transpose() * g).transpose());
//    ROS_INFO_STREAM("R0:\n" << Rs[0]);
}


void SlamSystem::insertFrame(ros::Time time, cv::Mat img )
{
    states[frame_count]->reset(time, img);
    states[frame_count]->initEdgePyramid();
    states[frame_count]->initDistanceTransformPyramid();
    states[frame_count]->state_id = frame_count ;

    states[frame_count]->frame.reset(
                new lsd_slam::Frame(frame_count, width, height, K, time.toSec(), img.data) ) ;

//    Ps[frame_count] = Ps[frame_count-1]
//            + Vs[frame_count-1]*imu_factors[frame_count]->pre_integration.sum_dt
//            - 0.5 * GRAVITY * SQ(imu_factors[frame_count]->pre_integration.sum_dt)
//            + Rs[frame_count-1]*imu_factors[frame_count]->alpha_c_k ;
//    Vs[frame_count] = Vs[frame_count-1]
//            - GRAVITY*imu_factors[frame_count]->pre_integration.sum_dt
//            +Rs[frame_count-1]*imu_factors[frame_count]->beta_c_k ;
//    Rs[frame_count] = Rs[frame_count-1]*imu_factors[frame_count]->R_k1_k ;

    if ( IMUorNot ){
        Ps[frame_count] = Ps[frame_count-1]
                + Vs[frame_count-1]*imu_factors[frame_count]->pre_integration.sum_dt
                - 0.5 * GRAVITY * SQ(imu_factors[frame_count]->pre_integration.sum_dt)
                + Rs[frame_count-1]*imu_factors[frame_count]->pre_integration.delta_p ;
        Vs[frame_count] = Vs[frame_count-1]
                - GRAVITY*imu_factors[frame_count]->pre_integration.sum_dt
                +Rs[frame_count-1]*imu_factors[frame_count]->pre_integration.delta_v ;
        Rs[frame_count] = Rs[frame_count-1]*imu_factors[frame_count]->pre_integration.delta_q ;

        Bas[frame_count] = Bas[frame_count-1] ;
        Bgs[frame_count] = Bgs[frame_count-1] ;
    }
    else{
        Ps[frame_count] = Ps[frame_count-1] ;
        Vs[frame_count] = Vs[frame_count-1] ;
        Rs[frame_count] = Rs[frame_count-1] ;
        Bas[frame_count] = Bas[frame_count-1] ;
        Bgs[frame_count] = Bgs[frame_count-1] ;
    }

    //    cout << "imu_factors[frame_count]->sum_dt " << imu_factors[frame_count]->pre_integration.sum_dt << "\n" ;
    //    cout << "imu_factors[frame_count]->delta_p " << imu_factors[frame_count]->pre_integration.delta_p  << "\n" ;
    //    cout << "imu_factors[frame_count]->delta_v "<< imu_factors[frame_count]->pre_integration.delta_v  << "\n" ;
    //    cout << "imu_factors[frame_count]->delta_q "<< imu_factors[frame_count]->pre_integration.delta_q.toRotationMatrix()  << "\n" ;
}

void SlamSystem::insertCameraLink(STATE* keyFrame, STATE* currentFrame,
                                  const Eigen::Matrix3d& R_k_2_c, const Eigen::Vector3d& T_k_2_c,
                                  const Eigen::Matrix<double, 6, 6> &lastestATA )
{
    CAMERALINK tmp;
    tmp.R_bi_2_bj = R_k_2_c ;
    tmp.T_bi_2_bj = T_k_2_c ;
    tmp.P_inv = lastestATA ;
    tmp.pState = currentFrame ;
    keyFrame->cameraLink.push_back(tmp);
    ROS_INFO("insert camera link %d %d", keyFrame->state_id, currentFrame->state_id ) ;
}

void SlamSystem::setReprojectionListRelateToLastestKeyFrameEdge()
{
    int trackFrameCnt = 0 ;
    int currentKFID = pKF->state_id ;
    for (int i = 0; i < currentKFID; i++)
    {
        if ( states[i]->keyFrameFlag == false
             || trackFrameCnt > 10
             ){
            continue;
        }
        if ( states[i]->edge_3d[0].cols() < MINIMUM_3DPOINTSFORTRACKING ||
             pKF->edge_3d[0].cols() < MINIMUM_3DPOINTSFORTRACKING )
        {
            continue ;
        }

        double closenessTH = 1.0 ;
        double distanceTH = closenessTH * 15 / (KFDistWeight*KFDistWeight);
        //double cosAngleTH = 1.0 - 0.25 * closenessTH ;

        //euclideanOverlapCheck
        double distFac = states[i]->iDepthMean ;
        Eigen::Vector3d dd = ( Ps[i] - Ps[currentKFID]) * distFac;
        if( dd.dot(dd) > distanceTH) continue;

        Matrix3d R_i_2_j ;
        Vector3d T_i_2_j ;
        Eigen::VectorXf epsilonVec;
        Eigen::MatrixXf reprojections;
        Eigen::VectorXf energyAtEachIteration;
        Eigen::Matrix3d cR_64 ;
        Eigen::Vector3d cT_64 ;
        int bestEnergyIndex = -1;
        float visibleRatio = 0.0;
        float error ;

        //check from current to ref
        R_i_2_j = Rs[i].transpose() * Rs[currentKFID] ;
        T_i_2_j = -Rs[i].transpose() * ( Ps[i] - Ps[currentKFID] ) ;

        cR_64 = R_i_2_j.transpose();
        cT_64 = -cR_64 * T_i_2_j ;
        for( int ith = optimizedPyramidLevel-1 ; ith >= optimizedPyramidLevel-1; ith-- )
        {
            error = dvoLoop.optimizeGaussianNewton( pKF, states[i], calib_par,
                                                    ith, gaussianNewtonTrackingIterationNum, cR_64, cT_64, energyAtEachIteration,
                                                    epsilonVec, reprojections, bestEnergyIndex, visibleRatio );
        }
        if ( visibleRatio < 0.7 || error > trustTrackingErrorThreshold ){
            continue ;
        }
        //ROS_WARN("pass first check") ;

        //check from ref to current
        R_i_2_j = Rs[currentKFID].transpose() * Rs[i] ;
        T_i_2_j = -Rs[currentKFID].transpose() * ( Ps[currentKFID] - Ps[i] ) ;
        cR_64 = R_i_2_j.transpose();
        cT_64 = -cR_64 * T_i_2_j ;
        for( int ith = optimizedPyramidLevel-1 ; ith >= optimizedPyramidLevel-1; ith-- )
        {
            error = dvoLoop.optimizeGaussianNewton( states[i], pKF, calib_par,
                                                    ith, gaussianNewtonTrackingIterationNum, cR_64, cT_64, energyAtEachIteration,
                                                    epsilonVec, reprojections, bestEnergyIndex, visibleRatio );
        }
        if ( visibleRatio < 0.7 || error > trustTrackingErrorThreshold ){
            continue ;
        }

        //ROS_WARN("pass second check") ;
        trackFrameCnt++ ;

        //Pass the cross check
        for( int ith = optimizedPyramidLevel-2 ; ith >= 0; ith-- )
        {
            error = dvoLoop.optimizeGaussianNewton( states[i], pKF, calib_par,
                                                    ith, gaussianNewtonTrackingIterationNum, cR_64, cT_64, energyAtEachIteration,
                                                    epsilonVec, reprojections, bestEnergyIndex, visibleRatio );
        }
        if ( visibleRatio < 0.7 || error > trustTrackingErrorThreshold ){
            continue ;
        }

#ifdef PROJECT_TO_IMU_CENTER
        Eigen::Matrix3d r_i_2_j = cR_64.transpose();
        Eigen::Vector3d t_i_2_j = -cR_64.transpose()*cT_64;
        Eigen::Matrix3d final_R = R_i_2_c.transpose()*r_i_2_j*R_i_2_c;
        Eigen::Vector3d final_T = R_i_2_c.transpose()*(r_i_2_j*T_i_2_c + t_i_2_j ) - R_i_2_c.transpose()*T_i_2_c ;
#else
        Eigen::Matrix3d final_R = cR_64.transpose();
        Eigen::Vector3d final_T = -cR_64.transpose()*cT_64;
#endif
        //ROS_WARN("[add link, from %d to %d]", slidingWindow[ref_id]->id(), current->id() ) ;
        Eigen::Matrix<double, 6, 6> ATA ;
        ATA.setIdentity() ;
        insertCameraLink( states[i], pKF,
                          final_R,
                          final_T,
                          ATA*visualWeight ) ;
        break ;
    }
}

void SlamSystem::optimizeWithEdgeAlignment(STATE* keyFrame, STATE* currentFrame,
                                           Matrix3d &R_i_2_j, Vector3d &t_i_2_j,
                                           CALIBRATION_PAR* cali, int level,
                                           double& ratio, double& aveError)
{
    Quaterniond q_i_2_j(R_i_2_j) ;
    double pose[7] = {t_i_2_j(0), t_i_2_j(1), t_i_2_j(2),
                      q_i_2_j.x(), q_i_2_j.y(), q_i_2_j.z(), q_i_2_j.w()} ;

    //    ROS_INFO("level=%d", level) ;
    //    // print final vals
    //    std::cout << "init values of q_init=[";
    //    for( int i=0 ; i<4 ; i++ )
    //        std::cout << pose[i+3] << " ";
    //    std::cout << "]';\n";

    //    std::cout << "init values of t_init=[";
    //    for( int i=0 ; i<3 ; i++ )
    //        std::cout << pose[i] << " ";
    //    std::cout << "]';\n";

    ceres::Problem problem;
    ceres::LossFunction *  loss_function = new ceres::HuberLoss(5.0);
    ceres::LocalParameterization* local_parameterization = new PoseLocalParameterization();
    problem.AddParameterBlock(pose, SIZE_POSE, local_parameterization);

    int width = currentFrame->distanceTransformEigen[level].cols();
    int height = currentFrame->distanceTransformEigen[level].rows();
    for( int i=0, sz = keyFrame->edge_3d[level].cols(); i < sz; i++ )
    {
        EdgeAlignmentFactor* f = new EdgeAlignmentFactor(keyFrame->edge_3d[level](0, i),
                                                         keyFrame->edge_3d[level](1, i),
                                                         keyFrame->edge_3d[level](2, i),
                                                         currentFrame->distanceTransformEigen[level],
                                                         currentFrame->distanceTransformGradientX[level],
                                                         currentFrame->distanceTransformGradientY[level],
                                                         cali->fx[level], cali->fy[level],
                                                         cali->cx[level], cali->cy[level],
                                                         width, height);

                problem.AddResidualBlock(f, NULL, pose);
    }

    // setting the solver
    ceres::Solver::Options options;
    options.max_num_iterations = 5;
    //    options.max_solver_time_in_seconds = 5;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = false;
    options.num_threads = 4;
    options.minimizer_type = ceres::TRUST_REGION;
    options.trust_region_strategy_type = ceres::DOGLEG;
    //options.dogleg_type = ceres::SUBSPACE_DOGLEG;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //    std::cout << summary.BriefReport() << "\n";
    //std::cout << summary.FullReport() << "\n";

    // print final vals
    //    std::cout << "final values of q_init=[";
    //    for( int i=0 ; i<4 ; i++ )
    //        std::cout << pose[i+3] << " ";
    //    std::cout << "]';\n";

    //    std::cout << "final values of t_init=[";
    //    for( int i=0 ; i<3 ; i++ )
    //        std::cout << pose[i] << " ";
    //    std::cout << "]';\n";

    Eigen::Quaterniond q_i_2_j_new(pose[6], pose[3], pose[4], pose[5]) ;
    R_i_2_j = q_i_2_j_new.toRotationMatrix() ;
    t_i_2_j << pose[0], pose[1], pose[2] ;

    Vector3d p2 ;
    int cnt = 0;
    double total_error = 0 ;
    const int nCols = currentFrame->distanceTransformEigen[level].cols()-3;
    const int nRows = currentFrame->distanceTransformEigen[level].rows()-3;
    for( int i=0, sz = keyFrame->edge_3d[level].cols(); i < sz; i++ )
    {
        p2 = R_i_2_j*keyFrame->edge_3d[level].col(i).cast<double>() + t_i_2_j ;

        double u_ = cali->fx[level] * p2(0) / p2(2) + cali->cx[level] ;
        double v_ = cali->fy[level] * p2(1) / p2(2) + cali->cy[level] ;
        if( u_ < 2 || u_ > nCols || v_ < 0 || v_ > nRows ){
            continue;
        }
        cnt++ ;

        total_error += getInterpolatedElementEigen(currentFrame->distanceTransformEigen[level], u_, v_) ; ;
    }
    ratio = (double)cnt/keyFrame->edge_3d[level].cols() ;
    aveError = total_error/cnt ;
}

void SlamSystem::setReprojectionListRelateToLastestKeyFrameEdgeCeres()
{
    int trackFrameCnt = 0 ;
    int currentKFID = pKF->state_id ;
    for (int i = 0; i < currentKFID; i++)
    {
        if ( states[i]->keyFrameFlag == false
             || trackFrameCnt > 10
             ){
            continue;
        }
        if ( states[i]->edge_3d[0].cols() < MINIMUM_3DPOINTSFORTRACKING ||
             pKF->edge_3d[0].cols() < MINIMUM_3DPOINTSFORTRACKING )
        {
            continue ;
        }

        double closenessTH = 1.0 ;
        double distanceTH = closenessTH * 15 / (KFDistWeight*KFDistWeight);
        //double cosAngleTH = 1.0 - 0.25 * closenessTH ;

        //euclideanOverlapCheck
        double distFac = states[i]->iDepthMean ;
        Eigen::Vector3d dd = ( Ps[i] - Ps[currentKFID]) * distFac;
        if( dd.dot(dd) > distanceTH) continue;

        Matrix3d R_i_2_j ;
        Vector3d T_i_2_j ;
        Eigen::Matrix3d cR_64 ;
        Eigen::Vector3d cT_64 ;
        double visibleRatio = 0.0;
        double error ;

        //check from current to ref
        R_i_2_j = Rs[i].transpose() * Rs[currentKFID] ;
        T_i_2_j = -Rs[i].transpose() * ( Ps[i] - Ps[currentKFID] ) ;
        cR_64 = R_i_2_j;
        cT_64 = T_i_2_j;
        for( int ith = optimizedPyramidLevel-1 ; ith >= optimizedPyramidLevel-1; ith-- )
        {
            optimizeWithEdgeAlignment(pKF, states[i],
                                      cR_64, cT_64,
                                      calib_par, ith,
                                      visibleRatio, error ) ;
        }
        if ( visibleRatio < 0.7 || error > trustTrackingErrorThreshold ){
            continue ;
        }
        //ROS_WARN("pass first check") ;

        //check from ref to current
        R_i_2_j = Rs[currentKFID].transpose() * Rs[i] ;
        T_i_2_j = -Rs[currentKFID].transpose() * ( Ps[currentKFID] - Ps[i] ) ;
        cR_64 = R_i_2_j;
        cT_64 = T_i_2_j;
        for( int ith = optimizedPyramidLevel-1 ; ith >= optimizedPyramidLevel-1; ith-- )
        {
            optimizeWithEdgeAlignment(states[i], pKF,
                                      cR_64, cT_64,
                                      calib_par, ith,
                                      visibleRatio, error ) ;
        }
        if ( visibleRatio < 0.7 || error > trustTrackingErrorThreshold ){
            continue ;
        }

        //ROS_WARN("pass second check") ;
        trackFrameCnt++ ;

        //Pass the cross check
        for( int ith = optimizedPyramidLevel-2 ; ith >= 0; ith-- )
        {
            optimizeWithEdgeAlignment(states[i], pKF,
                                      cR_64, cT_64,
                                      calib_par, ith,
                                      visibleRatio, error ) ;
        }
        if ( visibleRatio < 0.7 || error > trustTrackingErrorThreshold ){
            continue ;
        }

#ifdef PROJECT_TO_IMU_CENTER
        Eigen::Matrix3d r_i_2_j = cR_64;
        Eigen::Vector3d t_i_2_j = cT_64;
        Eigen::Matrix3d final_R = R_i_2_c.transpose()*r_i_2_j*R_i_2_c;
        Eigen::Vector3d final_T = R_i_2_c.transpose()*(r_i_2_j*T_i_2_c + t_i_2_j ) - R_i_2_c.transpose()*T_i_2_c ;
#else
        Eigen::Matrix3d final_R = cR_64.transpose();
        Eigen::Vector3d final_T = -cR_64.transpose()*cT_64;
#endif
        //ROS_WARN("[add link, from %d to %d]", slidingWindow[ref_id]->id(), current->id() ) ;
        Matrix<double, 6, 6> ATA ;
        ATA.setIdentity() ;
        insertCameraLink( states[i], pKF,
                          final_R,
                          final_T,
                          ATA*visualWeight ) ;
        break ;
    }
}

void SlamSystem::setReprojectionListRelateToLastestKeyFrameDense()
{
    int trackFrameCnt = 0 ;
    int currentKFID = pKF->state_id ;
    for (int i = 0; i < currentKFID; i++)
    {
        if ( states[i]->keyFrameFlag == false
             || trackFrameCnt > 10
             ){
            continue;
        }

        double closenessTH = 1.0 ;
        double distanceTH = closenessTH * 15 / (KFDistWeight*KFDistWeight);
        //double cosAngleTH = 1.0 - 0.25 * closenessTH ;

        //euclideanOverlapCheck
        double distFac = states[i]->iDepthMean ;
        Eigen::Vector3d dd = ( Ps[i] - Ps[currentKFID]) * distFac;
        if( dd.dot(dd) > distanceTH) continue;

        Matrix3d R_i_2_j ;
        Vector3d T_i_2_j ;
        SE3 c2f_init ;

        R_i_2_j = Rs[i].transpose() * Rs[currentKFID] ;
        T_i_2_j = -Rs[i].transpose() * ( Ps[i] - Ps[currentKFID] ) ;

        c2f_init.setRotationMatrix(R_i_2_j);
        c2f_init.translation() = T_i_2_j ;


        trackerConstraint->trackFrameOnPermaref(pKF->frame.get(), states[i]->frame.get(), c2f_init ) ;
        if ( trackerConstraint->trackingWasGood == false ){
            //ROS_WARN("first check fail") ;
            continue ;
        }
        //ROS_INFO("pass first check") ;

        //check from ref to current
        R_i_2_j = Rs[currentKFID].transpose() * Rs[i] ;
        T_i_2_j = -Rs[currentKFID].transpose() * ( Ps[currentKFID] - Ps[i] ) ;
        c2f_init.setRotationMatrix(R_i_2_j);
        c2f_init.translation() = T_i_2_j ;
        //printf("%d %d\n", states[i]->state_id, pKF->state_id ) ;
        trackerConstraint->trackFrameOnPermaref(states[i]->frame.get(), pKF->frame.get(), c2f_init ) ;
        if ( trackerConstraint->trackingWasGood == false ){
            //ROS_WARN("second check fail") ;
            continue ;
        }

        //ROS_INFO("pass second check") ;

        //Pass the cross check
        if (  trackingReferenceConstraint->keyframe !=  states[i]->frame.get() ){
            trackingReferenceConstraint->importFrame( states[i]->frame.get() );
        }

        SE3 RefToFrame = trackerConstraint->trackFrame( trackingReferenceConstraint, pKF->frame.get(),
                                                        c2f_init );

        //ROS_INFO("pass all check") ;
        trackFrameCnt++ ;

        float tracking_lastGoodPerTotal = trackerConstraint->lastGoodCount /
                (pKF->frame->width(SE3TRACKING_MIN_LEVEL)*pKF->frame->height(SE3TRACKING_MIN_LEVEL));
        Eigen::Vector3d dist = RefToFrame.translation() * states[i]->frame->meanIdepth;
        float minVal = 1.0f;
        float lastTrackingClosenessScore = getRefFrameScore(dist.dot(dist),
                                                            trackerConstraint->pointUsage,
                                                            KFDistWeight, KFUsageWeight);
        if ( trackerConstraint->trackingWasGood == false
             ||  tracking_lastGoodPerTotal < MIN_GOODPERALL_PIXEL
             || lastTrackingClosenessScore > minVal
             )
        {
            continue ;
        }

#ifdef PROJECT_TO_IMU_CENTER
        Eigen::Matrix3d r_i_2_j = RefToFrame.rotationMatrix().cast<double>();
        Eigen::Vector3d t_i_2_j = RefToFrame.translation().cast<double>();
        Eigen::Matrix3d final_R = R_i_2_c.transpose()*r_i_2_j*R_i_2_c;
        Eigen::Vector3d final_T = R_i_2_c.transpose()*(r_i_2_j*T_i_2_c + t_i_2_j ) - R_i_2_c.transpose()*T_i_2_c ;
#else
        Eigen::Matrix3d final_R = RefToFrame.rotationMatrix().cast<double>();
        Eigen::Vector3d final_T = RefToFrame.translation().cast<double>();
#endif
        //ROS_WARN("[add link, from %d to %d]", slidingWindow[ref_id]->id(), current->id() ) ;
        Matrix<double, 6, 6> ATA ;
        ATA.setIdentity() ;
        insertCameraLink( states[i], pKF,
                          final_R,
                          final_T,
                          ATA*visualWeight ) ;
        break ;
    }
}

void SlamSystem::processIMU(double dt, const Vector3d&linear_acceleration, const Vector3d &angular_velocity)
{
    if (!imu_factors[frame_count])
    {
        ROS_WARN("processIMU new frame_count = %d", frame_count ) ;
        imu_factors[frame_count] = new IMUFactor_t{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    imu_factors[frame_count]->push_back(dt, linear_acceleration, angular_velocity);

    acc_0 = linear_acceleration ;
    gyr_0 = angular_velocity ;
}

void SlamSystem::updateTrackingReference()
{
    if (  trackingReference->keyframe != currentKeyFrame.get() ){
        trackingReference->importFrame( currentKeyFrame.get() );
        currentKeyFrame->setPermaRef( trackingReference );
    }
}

void SlamSystem::para2eigen()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    ROS_INFO("origin R %lf %lf %lf", origin_R0.x(), origin_R0.y(), origin_R0.z());

    Vector3d origin_P0 = Ps[0];
    //ROS_INFO("origin P %f %f %f", origin_P0.x(), origin_P0.y(), origin_P0.z());

    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
            para_Pose[0][3],
            para_Pose[0][4],
            para_Pose[0][5]).normalized().toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();

    ROS_INFO("yaw_diff %lf", y_diff);

    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));

    for (int i = 0; i <= frame_count; i++)
    {
        //Vector3d tmp = Utility::R2ypr(Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix());
        //ROS_WARN("before R %d = %lf %lf %lf", i, tmp(0), tmp(1), tmp(2) ) ;

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                para_Pose[i][1] - para_Pose[0][1],
                para_Pose[i][2] - para_Pose[0][2]) +
                origin_P0;
        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                para_SpeedBias[i][1],
                para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                para_SpeedBias[i][4],
                para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                para_SpeedBias[i][7],
                para_SpeedBias[i][8]);

        //tmp = Utility::R2ypr(Rs[i]);
        //ROS_WARN("after R %d = %lf %lf %lf", i, tmp(0), tmp(1), tmp(2) ) ;
    }
    //        Vector3d cur_P0 = Ps[0];
    //        ROS_INFO("current P %f %f %f", cur_P0.x(), cur_P0.y(), cur_P0.z());

    Vector3d cur_R0 = Utility::R2ypr(Rs[0]);
    ROS_INFO("current R %lf %lf %lf", cur_R0.x(), cur_R0.y(), cur_R0.z());

#if 1
    ROS_ASSERT((origin_P0 - cur_P0).norm() < 1e-6);
    ROS_ASSERT((origin_R0.x() - cur_R0.x()) < 1e-6);
#endif
}

void SlamSystem::moveData()
{
    if ( states[frame_count-1]->keyFrameFlag && frame_count == WINDOW_SIZE )
    {//marginalize old
        for (int i = 0; i < frame_count; i++)
        {
            Rs[i].swap(Rs[i + 1]);
            //IMUFactor_t * tmpIMU = imu_factors[i] ;
            //imu_factors[i] = imu_factors[i+1] ;
            //imu_factors[i+1] = tmpIMU ;
            std::swap(imu_factors[i], imu_factors[i + 1]);
            Ps[i].swap(Ps[i + 1]);
            Vs[i].swap(Vs[i + 1]);
            Bas[i].swap(Bas[i + 1]);
            Bgs[i].swap(Bgs[i + 1]);

            std::swap(states[i], states[i+1] ) ;
            //STATE* tmp = states[i] ;
            //states[i] = states[i+1] ;
            //states[i+1] = tmp ;

            states[i]->state_id = i ;
        }
        states[frame_count]->state_id = frame_count ;
        Ps[frame_count] = Ps[frame_count - 1];
        Vs[frame_count] = Vs[frame_count - 1];
        Rs[frame_count] = Rs[frame_count - 1];
        Bas[frame_count] = Bas[frame_count - 1];
        Bgs[frame_count] = Bgs[frame_count - 1];

        if ( imu_factors[frame_count] != nullptr ){
            delete imu_factors[frame_count];
        }
        imu_factors[frame_count] = nullptr ;

        frame_count-- ;
    }
    else if ( states[frame_count-1]->keyFrameFlag == false
              //&& frame_count == WINDOW_SIZE
              )
    {//marginalize second newest
        if ( frontMarginalization )
        {
            if ( imu_factors[frame_count - 1] != nullptr ){
                delete imu_factors[frame_count - 1] ;
            }
            imu_factors[frame_count - 1] = nullptr ;
        }
        else//drop
        {
            unsigned int sz = imu_factors[frame_count]->pre_integration.dt_buf.size() ;
            for (unsigned int i = 0; i < sz; i++)
            {
                imu_factors[frame_count - 1]->push_back(imu_factors[frame_count]->pre_integration.dt_buf[i],
                                                        imu_factors[frame_count]->pre_integration.acc_buf[i],
                                                        imu_factors[frame_count]->pre_integration.gyr_buf[i]);
            }
        }

        Ps[frame_count - 1] = Ps[frame_count];
        Vs[frame_count - 1] = Vs[frame_count];
        Rs[frame_count - 1] = Rs[frame_count];
        Bas[frame_count - 1] = Bas[frame_count];
        Bgs[frame_count - 1] = Bgs[frame_count];

        //erase camera link
        for (int i = 0; i < frame_count; i++)
        {
            for(std::list<CAMERALINK>::iterator iter = states[i]->cameraLink.begin();
                iter != states[i]->cameraLink.end(); )
            {
                if ( iter->pState == states[frame_count-1] ){
                    //printf("remove link %d %d\n", i, states[frame_count-1]->state_id ) ;
                    iter = states[i]->cameraLink.erase(iter) ;
                }
                else{
                    iter++;
                }
            }
        }
        //STATE* tmp = states[frame_count-1] ;
        //states[frame_count-1] = states[frame_count] ;
        //states[frame_count] = tmp ;
        std::swap( states[frame_count-1],  states[frame_count]) ;
        states[frame_count-1]->state_id = frame_count-1;

        if ( imu_factors[frame_count] != nullptr ){
            delete imu_factors[frame_count];
        }
        imu_factors[frame_count] = nullptr ;

        frame_count-- ;
    }
}

void SlamSystem::marginalization()
{
    if ( states[frame_count-1]->keyFrameFlag && frame_count == WINDOW_SIZE )
    {//marginalize old
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor();
        eigen2para();
        ROS_INFO("marginalise oldest") ;
        if (last_marginalization_factor)//last marginalization factor
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                        last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(last_marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_factor->addResidualBlockInfo(residual_block_info);
        }

        if ( IMUorNot )
        {//added imu factor
            if ( imu_factors[1] != nullptr )
            {
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factors[1], NULL,
                        vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                vector<int>{0, 1});
                marginalization_factor->addResidualBlockInfo(residual_block_info);
            }
        }

        {//added visual factor
            for( std::list<CAMERALINK>::iterator iter = states[0]->cameraLink.begin();
                 iter != states[0]->cameraLink.end(); iter++ )
            {
                int linkID = iter->pState->state_id ;
                RelativePoseFactor *f = new RelativePoseFactor(iter->R_bi_2_bj, iter->T_bi_2_bj, iter->P_inv);
                ResidualBlockInfo * residual_block_info =
                        new ResidualBlockInfo(f, NULL, vector<double *>{para_Pose[0], para_Pose[linkID]}, vector<int>{0});
                marginalization_factor->addResidualBlockInfo(residual_block_info);
            }
        }

        TicToc t_pre_margin;
        ROS_INFO("begin pre marginalization");
        marginalization_factor->preMarginalize();
        ROS_INFO("end pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        ROS_INFO("begin marginalization");
        marginalization_factor->marginalize();
        ROS_INFO("end marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= frame_count; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if ( IMUorNot ){
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
            }
        }

        vector<double *> parameter_blocks = marginalization_factor->getParameterBlocks(addr_shift);
        last_marginalization_factor = marginalization_factor;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else if ( states[frame_count-1]->keyFrameFlag == false
              //&& frame_count == WINDOW_SIZE
              )
    {//marginalize new -- drop
        //        if ( last_marginalization_factor
        //              // && std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[frame_count - 1])
        //             )
        {

            MarginalizationFactor *marginalization_factor = new MarginalizationFactor();
            eigen2para();
            ROS_INFO("marginalise second newest") ;
            if (last_marginalization_factor)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    if ( IMUorNot ){
                        ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[frame_count - 1]);
                    }
                    if (last_marginalization_parameter_blocks[i] == para_Pose[frame_count - 1])
                        drop_set.push_back(i);
                }
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(last_marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_factor->addResidualBlockInfo(residual_block_info);
            }

            if ( frontMarginalization )
            {
                //add imu factor
                if ( IMUorNot && imu_factors[frame_count-1] != nullptr )
                {
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factors[frame_count-1], NULL,
                            vector<double *>{para_Pose[frame_count-2], para_SpeedBias[frame_count-2],
                                para_Pose[frame_count-1], para_SpeedBias[frame_count-1]},
                    vector<int>{2, 3});
                    marginalization_factor->addResidualBlockInfo(residual_block_info);

                    printf("[front Margin] add IMU %d\n", frame_count-1 ) ;
                }

                if ( IMUorNot && imu_factors[frame_count] != nullptr )
                {
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factors[frame_count], NULL,
                                                                                   vector<double *>{para_Pose[frame_count-1], para_SpeedBias[frame_count-1],
                                                                                                    para_Pose[frame_count], para_SpeedBias[frame_count]},
                                                                                   vector<int>{0, 1});
                    marginalization_factor->addResidualBlockInfo(residual_block_info);

                    printf("[front Margin] add IMU %d\n", frame_count ) ;
                }

                //add visual factor
                for( std::list<CAMERALINK>::iterator iter = lastKF->cameraLink.begin();
                     iter != lastKF->cameraLink.end(); iter++ )
                {
                    if ( iter->pState != states[frame_count-1] ){
                        continue ;
                    }
                    RelativePoseFactor *f = new RelativePoseFactor(iter->R_bi_2_bj, iter->T_bi_2_bj, iter->P_inv);
                    ResidualBlockInfo * residual_block_info =
                            new ResidualBlockInfo(f, NULL, vector<double *>{para_Pose[lastKF->state_id],
                                                                            para_Pose[frame_count-1]}, vector<int>{1});
                    marginalization_factor->addResidualBlockInfo(residual_block_info);

                    printf("[front Margin] add Viual %d %d\n", lastKF->state_id, frame_count-1 ) ;
                }
            }

            //            if ( last_marginalization_factor
            //                 && std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[frame_count - 1])
            //            )
            if ( frontMarginalization ||
                 (last_marginalization_factor && std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[frame_count - 1]))
                 )
            {
                TicToc t_pre_margin;
                ROS_INFO("begin marginalization");
                marginalization_factor->preMarginalize();
                ROS_INFO("end pre marginalization, %f ms", t_pre_margin.toc());

                TicToc t_margin;
                ROS_INFO("begin marginalization");
                marginalization_factor->marginalize();
                ROS_INFO("end marginalization, %f ms", t_margin.toc());

                std::unordered_map<long, double *> addr_shift;
                for (int i = 0; i <= frame_count; i++)
                {
                    if (i == frame_count - 1)
                        continue;
                    else if (i == frame_count)
                    {
                        addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                        if ( IMUorNot ){
                            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                        }
                    }
                    else
                    {
                        addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                        if ( IMUorNot ){
                            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                        }
                    }
                }

                vector<double *> parameter_blocks = marginalization_factor->getParameterBlocks(addr_shift);
                last_marginalization_factor = marginalization_factor;
                last_marginalization_parameter_blocks = parameter_blocks;
            }
        }
    }
}

void SlamSystem::solve_ceres()
{
#define DEBUG_CERES
    //#define DEBUG_CERES
    //init ceres
    while ( ceres_frame_count < frame_count )
    {
        ceres_frame_count++ ;
        PoseLocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[ceres_frame_count], SIZE_POSE, local_parameterization);
        if ( IMUorNot ){
            ceres::SubsetParameterization* sub_par =
                    new ceres::SubsetParameterization(SIZE_SPEEDBIAS, vector<int>{3, 4, 5, 6, 7, 8}) ;
            problem.AddParameterBlock(para_SpeedBias[ceres_frame_count], SIZE_SPEEDBIAS);
            //problem.AddParameterBlock(para_SpeedBias[ceres_frame_count], SIZE_SPEEDBIAS, sub_par);
        }
        if ( ceres_frame_count == 0 && IMUorNot == false ){
            problem.SetParameterBlockConstant(para_Pose[ceres_frame_count]);
        }
    }
    eigen2para();

    IMUFactor_t::sum_t = 0.0;

    double sum_error_ceres = 0;

    {//marginalization factor
        double m_sum = 0;
        if (last_marginalization_factor)
        {
            problem.AddResidualBlock(last_marginalization_factor, NULL,
                                     last_marginalization_parameter_blocks);

#ifdef DEBUG_CERES
            double **para = new double *[last_marginalization_parameter_blocks.size()];
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                para[i] = last_marginalization_parameter_blocks[i];
            double *res = new double[last_marginalization_factor->num_residuals()];

            last_marginalization_factor->Evaluate(para, res, NULL);
            for (int i = 0; i < last_marginalization_factor->num_residuals(); i++){
                m_sum += res[i] * res[i];
            }
#endif
        }
        ROS_INFO("marginalization error: %f", m_sum);
        sum_error_ceres += m_sum;
    }

    if ( IMUorNot )
    {//imu factor
        double i_sum = 0.0;
        printf("KF_INFO ") ;
        for( int i=0; i <= frame_count; i++ ){
            printf("%d ", states[i]->keyFrameFlag ) ;
        }
        printf("\n") ;
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if ( imu_factors[j] == nullptr ){
                continue ;
            }
            problem.AddResidualBlock(imu_factors[j], NULL,
                                     para_Pose[i], para_SpeedBias[i],
                                     para_Pose[j], para_SpeedBias[j]);
            //define DEBUG_CERES
#ifdef DEBUG_CERES
            double **para = new double *[4];
            para[0] = para_Pose[i];
            para[1] = para_SpeedBias[i];
            para[2] = para_Pose[j];
            para[3] = para_SpeedBias[j];
            double *tmp_r = new double[15];
            imu_factors[j]->Evaluate(para, tmp_r, NULL);
            double tmp_sum = 0.0;
            for (int j = 0; j < 15; j++)
            {
                tmp_sum += tmp_r[j] * tmp_r[j];
                //printf("%f ", tmp_r[j] * tmp_r[j]);
            }
            //puts("");
            i_sum += tmp_sum;
            //imu_factors[j]->checkJacobian(para);
#endif
        }
        //puts("");
        ROS_INFO("imu error: %f", i_sum);
        sum_error_ceres += i_sum;
    }

    //ROS_INFO("relative pose factor") ;
    {//relative pose factor
        double v_sum = 0;
        for( int i= 0 ; i < frame_count; i++ )
        {
            for( std::list<CAMERALINK>::iterator iter = states[i]->cameraLink.begin();
                 iter != states[i]->cameraLink.end(); iter++ )
            {
                int linkID = iter->pState->state_id ;
                //printf("[VF] from=%d to=%d\n", i, linkID ) ;
                //                if ( frame_count == 4 ){
                //                    std::cout << iter->R_bi_2_bj << "\n" ;
                //                    std::cout << iter->T_bi_2_bj.transpose() << "\n" ;
                //                    std::cout << iter->P_inv << "\n" ;
                //                }

                RelativePoseFactor *f = new RelativePoseFactor(iter->R_bi_2_bj, iter->T_bi_2_bj, iter->P_inv);
                problem.AddResidualBlock(f, NULL, para_Pose[i], para_Pose[linkID] );

#ifdef DEBUG_CERES
                double **para = new double *[2];
                para[0] = para_Pose[i];
                para[1] =  para_Pose[linkID];

                double *tmp_r = new double[6];
                f->Evaluate(para, tmp_r, NULL);
                double tmp_sum = 0.0;
                for (int j = 0; j < 6; j++)
                {
                    tmp_sum += tmp_r[j] * tmp_r[j];
                    //printf("%f ", tmp_r[j] * tmp_r[j]);
                }
                //puts("");
                v_sum += tmp_sum;
#endif
                //ROS_INFO("visual error: %f", v_sum);
            }
        }
        sum_error_ceres += v_sum;
        ROS_INFO("visual error: %f", v_sum);
    }
    ROS_INFO("total error: %f", sum_error_ceres);

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.num_threads = 6;
    options.trust_region_strategy_type = ceres::DOGLEG;
    //options.use_explicit_schur_complement = true;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;
    //options.use_nonmonotonic_steps = true;
    options.max_solver_time_in_seconds = SOLVER_TIME;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.BriefReport() << std::endl ;
    //cout << summary.FullReport() << endl;
    para2eigen();


    sum_error_ceres = 0 ;
    {//marginalization factor
        double m_sum = 0;
        if (last_marginalization_factor)
        {
#ifdef DEBUG_CERES
            double **para = new double *[last_marginalization_parameter_blocks.size()];
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                para[i] = last_marginalization_parameter_blocks[i];
            double *res = new double[last_marginalization_factor->num_residuals()];

            last_marginalization_factor->Evaluate(para, res, NULL);
            for (int i = 0; i < last_marginalization_factor->num_residuals(); i++){
                m_sum += res[i] * res[i];
            }
#endif
        }
        ROS_INFO("marginalization error: %f", m_sum);
        sum_error_ceres += m_sum;
    }

    if ( IMUorNot )
    {//imu factor
        double i_sum = 0.0;
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if ( imu_factors[j] == nullptr ){
                continue ;
            }
#ifdef DEBUG_CERES
            double **para = new double *[4];
            para[0] = para_Pose[i];
            para[1] = para_SpeedBias[i];
            para[2] = para_Pose[j];
            para[3] = para_SpeedBias[j];
            double *tmp_r = new double[15];
            imu_factors[j]->Evaluate(para, tmp_r, NULL);
            double tmp_sum = 0.0;
            for (int j = 0; j < 15; j++)
            {
                tmp_sum += tmp_r[j] * tmp_r[j];
                //printf("%f ", tmp_r[j] * tmp_r[j]);
            }
            //puts("");
            i_sum += tmp_sum;
            //imu_factors[j]->checkJacobian(para);
#endif
        }
        //puts("");
        ROS_INFO("imu error: %f", i_sum);
        sum_error_ceres += i_sum;
    }

    ROS_INFO("relative pose factor") ;
    {//relative pose factor
        double v_sum = 0;
        for( int i= 0 ; i < frame_count; i++ )
        {
            for( std::list<CAMERALINK>::iterator iter = states[i]->cameraLink.begin();
                 iter != states[i]->cameraLink.end(); iter++ )
            {
                int linkID = iter->pState->state_id ;
                //printf("[VF] from=%d to=%d\n", i, linkID ) ;
                //                if ( frame_count == 4 ){
                //                    std::cout << iter->R_bi_2_bj << "\n" ;
                //                    std::cout << iter->T_bi_2_bj.transpose() << "\n" ;
                //                    std::cout << iter->P_inv << "\n" ;
                //                }

                RelativePoseFactor *f = new RelativePoseFactor(iter->R_bi_2_bj, iter->T_bi_2_bj, iter->P_inv);
#ifdef DEBUG_CERES
                double **para = new double *[2];
                para[0] = para_Pose[i];
                para[1] =  para_Pose[linkID];

                double *tmp_r = new double[6];
                f->Evaluate(para, tmp_r, NULL);
                double tmp_sum = 0.0;
                for (int j = 0; j < 6; j++)
                {
                    tmp_sum += tmp_r[j] * tmp_r[j];
                    //printf("%f ", tmp_r[j] * tmp_r[j]);
                }
                //puts("");
                v_sum += tmp_sum;
#endif
                //ROS_INFO("visual error: %f", v_sum);

            }
        }
        sum_error_ceres += v_sum;
        ROS_INFO("visual error: %f", v_sum);
    }
    ROS_INFO("total error: %f", sum_error_ceres);


    vector<ceres::ResidualBlockId> residual_set;
    problem.GetResidualBlocks(&residual_set);
    for (auto it : residual_set){
        problem.RemoveResidualBlock(it);
    }

    if ( IMUorNot ){
        marginalization() ;
    }
    moveData() ;
}

void SlamSystem::eigen2para()
{
    for (int i = 0; i <= frame_count; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        para_Pose[i][3] = Quaterniond(Rs[i]).x();
        para_Pose[i][4] = Quaterniond(Rs[i]).y();
        para_Pose[i][5] = Quaterniond(Rs[i]).z();
        para_Pose[i][6] = Quaterniond(Rs[i]).w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
}

void SlamSystem::trackFrameEdge(cv::Mat image0, cv::Mat image1, unsigned int frameID,
                                ros::Time imageTimeStamp, Eigen::Matrix3d deltaR)
{
    //cR_64 = cR_64*deltaR ;
    cR_64 = deltaR.transpose()*cR_64 ;
    double bestTotalEpsilon = 100000000000.0;

    //if ( currentKeyFrame->numOfposData[0] < MINIMUM_3DPOINTSFORTRACKING )
    if ( pKF->edge_3d[0].cols() < MINIMUM_3DPOINTSFORTRACKING )
    {//not enough points for tracking
        createNewKeyFrame = true ;
    }
    else
    {
        createNewKeyFrame = false ;
        //        Eigen::Matrix3d backup_cR_64 = cR_64 ;
        //        Eigen::Vector3d backup_cT_64 = cT_64 ;

        currentFrame.reset(imageTimeStamp, image0);

        //double t = (double)cvGetTickCount()  ;
        currentFrame.initEdgePyramid();
        //ROS_WARN("Edge detection time: %f", ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000) );



        //        std::shared_ptr<lsd_slam::Frame> trackingNewFrame(
        //                    new lsd_slam::Frame( frameID, width, height, K, imageTimeStamp.toSec(), image0.data ) );



        //ROS_WARN("keyFrame=%d currentFrame=%d\n", currentKeyFrame->edgeNum[0], currentFrame.edgeNum[0] ) ;
        for( int i = 0 ; i < optimizedPyramidLevel-1 ; i++ )
        {
            if ( fabs(pKF->edgeNum[i]-currentFrame.edgeNum[i])/
                 double( min(pKF->edgeNum[i], currentFrame.edgeNum[i]) ) > edgeProportion )
            {
                //ROS_WARN("keyFrame=%d currentFrame=%d", currentKeyFrame->edgeNum[i], currentFrame.edgeNum[i] ) ;
                createNewKeyFrame = true ;
            }
        }

        if ( createNewKeyFrame ){
            ;
        }
        else
        {
            //t = (double)cvGetTickCount()  ;
            currentFrame.initDistanceTransformPyramid();
            //ROS_WARN("distance transform time: %f", ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000) );

            //            if ( enable_crossCheckTracking )
            //            {
            //                cv::Mat disparity, iDepth, iVar ;
            //                bm_(image0, image1, disparity, CV_32F);
            //                calculateInvDepthImage(disparity, iDepth, iVar, 0.11, calib_par->fx[0] );
            //                currentFrame.insertDepth(iDepth, iVar );
            //                currentFrame.initPointsPyramid(calib_par);
            //            }



            //track
            //printf("currentKeyFrame = %d 3d() = %d\n", currentKeyFrame->id(), currentKeyFrame->edge_3d[0].cols() ) ;

            for( int i = optimizedPyramidLevel-1 ; i >= 0; i-- )
            {
                optimizeWithEdgeAlignment(pKF, &currentFrame,
                                          cR_64, cT_64,
                                          calib_par, i,
                                          visibleRatio, bestTotalEpsilon );

                //                bestTotalEpsilon = dvo.optimizeGaussianNewton( pKF, &currentFrame, calib_par,
                //                                                               i, gaussianNewtonTrackingIterationNum, cR_64, cT_64, energyAtEachIteration,
                //                                                               epsilonVec, reprojections, bestEnergyIndex, visibleRatio );
            }
            if ( visualizeTrackingDebug )
            {
                //dvo.visualizeEnergyProgress( energyAtEachIteration, bestEnergyIndex, 2 );
                //Eigen::MatrixXi reprojectedMask = Eigen::MatrixXi::Zero( currentFrame.image[0].rows ,
                        //currentFrame.image[0].cols );
                //dvo.cordList_2_mask(reprojections, reprojectedMask);
                //Eigen::MatrixXf imageDisplay ;
                //cv::cv2eigen(currentFrame.image[0], imageDisplay) ;
                //dvo.visualizeDistanceResidueHeatMap( currentFrame.image[0], reprojectedMask, currentFrame.distanceTransformEigen[0] );

                cv::Mat displayDT ;
                cv::imshow("currentImage", currentFrame.image[0] ) ;
                cv::imshow("keyFrame-edge", pKF->edge[0] ) ;
                cv::moveWindow("keyFrame-edge", 0, 900 ) ;
                cv::imshow("currentFrame-edge", currentFrame.edge[0] ) ;
                cv::moveWindow("currentFrame-edge", 0, 280 ) ;
                cv::normalize( currentFrame.distanceTransformMap[0], displayDT, 0.0, 1.0, cv::NORM_MINMAX);
                cv::imshow("currentFrame-distance transform", displayDT ) ;
                cv::moveWindow("currentFrame-distance transform", 0, 550 ) ;
                cv::waitKey(1) ;
            }

            // Keyframe selection
            if ( visibleRatio < 0.7 || bestTotalEpsilon > trustTrackingErrorThreshold  ){
                createNewKeyFrame = true;
                nTrackFrame = 0 ;
            }
            else
            {
                createNewKeyFrame = false ;
                nTrackFrame++;
            }
            //ROS_INFO("%f\n", bestTotalEpsilon ) ;
        }
    }

    if ( lock_densetracking ){
        return ;
    }

    frameInfoList_mtx.lock();
    int tmpTail = frameInfoListTail+1 ;
    if ( tmpTail >= frameInfoListSize ){
        tmpTail -= frameInfoListSize;
    }
    FRAMEINFO& tmpFrameInfo = frameInfoList[tmpTail] ;
    tmpFrameInfo.t = imageTimeStamp ;

    // #if 1
    // //    tmpFrameInfo.R_k_2_c = RefToFrame.rotationMatrix().cast<double>();
    // //    tmpFrameInfo.T_k_2_c = RefToFrame.translation().cast<double>();
    //     tmpFrameInfo.R_k_2_c = cR_64.transpose();
    //     tmpFrameInfo.T_k_2_c = -cR_64.transpose()*cT_64;
    // #else
    //     Eigen::Matrix3d r_k_2_c = RefToFrame.rotationMatrix().cast<double>();
    //     Eigen::Vector3d t_k_2_c = RefToFrame.translation().cast<double>();
    //     tmpFrameInfo.R_k_2_c = R_i_2_c.transpose()*r_k_2_c*R_i_2_c;
    //     tmpFrameInfo.T_k_2_c = R_i_2_c.transpose()*(r_k_2_c*T_i_2_c + t_k_2_c ) - R_i_2_c.transpose()*T_i_2_c ;
    // #endif

#ifdef PROJECT_TO_IMU_CENTER
    Eigen::Matrix3d r_k_2_c = cR_64;
    Eigen::Vector3d t_k_2_c = cT_64;
    //Eigen::Matrix3d r_k_2_c = cR_64.transpose();
    //Eigen::Vector3d t_k_2_c = -cR_64.transpose()*cT_64;
    tmpFrameInfo.R_k_2_c = R_i_2_c.transpose()*r_k_2_c*R_i_2_c;
    tmpFrameInfo.T_k_2_c = R_i_2_c.transpose()*(r_k_2_c*T_i_2_c + t_k_2_c ) - R_i_2_c.transpose()*T_i_2_c ;
#else
    tmpFrameInfo.R_k_2_c = cR_64.transpose();
    tmpFrameInfo.T_k_2_c = -cR_64.transpose()*cT_64;
#endif

    //tmpFrameInfo.trust = tracker->trackingWasGood ;
    //tmpFrameInfo.trust = !createNewKeyFrame ;
    tmpFrameInfo.trust = true ;
//    if( bestTotalEpsilon > trustTrackingErrorThreshold ){
//        tmpFrameInfo.trust = false ;
//    }
    tmpFrameInfo.keyFrameFlag = createNewKeyFrame ;
    tmpFrameInfo.lastestATA.setIdentity() ;
    tmpFrameInfo.lastestATA *= visualWeight ;
    frameInfoListTail = tmpTail ;
    frameInfoList_mtx.unlock();
    if ( createNewKeyFrame == true ){
        tracking_mtx.lock();
        lock_densetracking = true ;
        tracking_mtx.unlock();
    }

    //    pubOdometry(-(tmpFrameInfo.R_k_2_c).transpose()*tmpFrameInfo.T_k_2_c,
    //                Eigen::Vector3d::Zero(),
    //                tmpFrameInfo.R_k_2_c.transpose(),
    //                pub_odometry, pub_pose, 0,
    //                Eigen::Matrix3d::Identity(), false, imageTimeStamp
    //                );
}

void SlamSystem::trackFrameDense(cv::Mat image0, unsigned int frameID,
                                 ros::Time imageTimeStamp, Eigen::Matrix3d deltaR)
{
    std::shared_ptr<lsd_slam::Frame> trackingNewFrame(
                new lsd_slam::Frame( frameID, width, height, K, imageTimeStamp.toSec(), image0.data ) );

    //initial guess
    SE3 RefToFrame_initialEstimate ;
    RefToFrame_initialEstimate.setRotationMatrix(  deltaR.transpose()*RefToFrame.rotationMatrix() );
    RefToFrame_initialEstimate.translation() =
            deltaR.transpose()*RefToFrame.translation() ;

    //track
    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);
    RefToFrame = tracker->trackFrame( trackingReference, trackingNewFrame.get(),
                                      RefToFrame_initialEstimate );
    gettimeofday(&tv_end, NULL);

    //    Eigen::Matrix3d R_k_2_c = RefToFrame.rotationMatrix();
    //    Eigen::Vector3d T_k_2_c = RefToFrame.translation();
    //    Matrix3d R_bk1_2_b0 = trackingReference->keyframe->R_bk_2_b0 * R_k_2_c.transpose();
    //    Vector3d T_bk1_2_b0 = trackingReference->keyframe->T_bk_2_b0 + R_bk1_2_b0*T_k_2_c ;
    //    pubOdometry(-T_bk1_2_b0, R_bk1_2_b0, pub_odometry, pub_pose );
    //    pubPath(-T_bk1_2_b0, path_line, pub_path );

    //debug information
    //msTrackFrame = 0.9*msTrackFrame + 0.1*((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
    msTrackFrame = (tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f ;
    printf("msTrackFrame = %0.f\n", msTrackFrame ) ;
    nTrackFrame++;
    tracking_lastResidual = tracker->lastResidual;
    tracking_lastUsage = tracker->pointUsage;
    tracking_lastGoodPerBad = tracker->lastGoodCount / (tracker->lastGoodCount + tracker->lastBadCount);
    tracking_lastGoodPerTotal = tracker->lastGoodCount / (trackingNewFrame->width(SE3TRACKING_MIN_LEVEL)*trackingNewFrame->height(SE3TRACKING_MIN_LEVEL));

    //    geometry_msgs::Vector3 v_pub ;
    //    Vector3d translation = RefToFrame.translation() ;
    //    v_pub.x = translation(0) ;
    //    v_pub.y = translation(1) ;
    //    v_pub.z = translation(2) ;
    //    pub_denseTracking.publish( v_pub ) ;

    // Keyframe selection
    createNewKeyFrame = false ;
    //printf("tracking_lastGoodPerTotal = %f\n", tracking_lastGoodPerTotal ) ;
    if ( trackingReference->keyframe->numFramesTrackedOnThis > MIN_NUM_MAPPED )
    {
        Eigen::Vector3d dist = RefToFrame.translation() * currentKeyFrame->meanIdepth;
        float minVal = 1.0f;

        lastTrackingClosenessScore = getRefFrameScore(dist.dot(dist), tracker->pointUsage, KFDistWeight, KFUsageWeight);
        if (lastTrackingClosenessScore > minVal || tracker->trackingWasGood == false
                || tracking_lastGoodPerTotal < MIN_GOODPERALL_PIXEL
                )
        {
            createNewKeyFrame = true;

            // if(enablePrintDebugInfo && printKeyframeSelectionInfo)
            //     printf("[insert KF] dist %.3f + usage %.3f = %.3f > 1\n", dist.dot(dist), tracker->pointUsage, lastTrackingClosenessScore );
        }
        else
        {
            //	if(enablePrintDebugInfo && printKeyframeSelectionInfo)
            //       printf("SKIPPD %d on %d! dist %.3f + usage %.3f = %.3f < 1\n",trackingNewFrame->id(),trackingNewFrame->getTrackingParent()->id(), dist.dot(dist), tracker->pointUsage, lastTrackingClosenessScore );
        }
    }
    if ( tracker->diverged ){
        createNewKeyFrame = true ;
    }
    frameInfoList_mtx.lock();
    int tmpTail = frameInfoListTail+1 ;
    if ( tmpTail >= frameInfoListSize ){
        tmpTail -= frameInfoListSize;
    }
    FRAMEINFO& tmpFrameInfo = frameInfoList[tmpTail] ;
    tmpFrameInfo.t = imageTimeStamp ;

#ifdef PROJECT_TO_IMU_CENTER
    Eigen::Matrix3d r_k_2_c = RefToFrame.rotationMatrix().cast<double>();
    Eigen::Vector3d t_k_2_c = RefToFrame.translation().cast<double>();
    tmpFrameInfo.R_k_2_c = R_i_2_c.transpose()*r_k_2_c*R_i_2_c;
    tmpFrameInfo.T_k_2_c = R_i_2_c.transpose()*(r_k_2_c*T_i_2_c + t_k_2_c ) - R_i_2_c.transpose()*T_i_2_c ;
#else
    tmpFrameInfo.R_k_2_c = RefToFrame.rotationMatrix().cast<double>();
    tmpFrameInfo.T_k_2_c = RefToFrame.translation().cast<double>();
#endif

    //    ROS_WARN("trackFrame = ") ;
    //    std::cout << tmpFrameInfo.T_k_2_c.transpose() << std::endl;

    tmpFrameInfo.trust = tracker->trackingWasGood ;
    tmpFrameInfo.keyFrameFlag = createNewKeyFrame ;
    tmpFrameInfo.lastestATA = MatrixXd::Identity(6, 6)*visualWeight ;
    frameInfoListTail = tmpTail ;
    frameInfoList_mtx.unlock();

    if ( createNewKeyFrame == true ){
        tracking_mtx.lock();
        lock_densetracking = true ;
        tracking_mtx.unlock();
    }
}
