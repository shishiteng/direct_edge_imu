#include "LiveSLAMWrapper.h"
#include <vector>
#include <list>
#include <iostream>
#include "globalFuncs.h"
#include "SlamSystem.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/PointCloud2.h"


LiveSLAMWrapper::LiveSLAMWrapper(std::string packagePath, ros::NodeHandle& _nh, CALIBRATION_PAR* calib_par)
{
    fx = calib_par->fx[0];
    fy = calib_par->fy[0];
    cx = calib_par->cx[0];
    cy = calib_par->cy[0];
    width = calib_par->width[0];
    height = calib_par->height[0];
    R_i_2_c = calib_par->R_i_2_c;
    T_i_2_c = calib_par->T_i_2_c;
    nh = _nh ;
    initRosPub();
    cali = calib_par ;

    isInitialized = false;
    Sophus::Matrix3f K_sophus;
    K_sophus << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;

    //R_vi_2_odometry << 0, 0, 1, -1, 0, 0, 0, -1, 0 ;
    R_vi_2_odometry.setZero() ;
    R_vi_2_odometry(0, 2) = 1.0 ;
    R_vi_2_odometry(1, 1) = 1.0 ;
    R_vi_2_odometry(2, 0) = 1.0 ;

    outFileName = packagePath+"estimated_poses.txt";
    //outFileName = packagePath+"angular_volcity.txt";
    outFile.open(outFileName);

	// make Odometry
    Odometry = new SlamSystem(width, height, K_sophus, calib_par );
    Odometry->R_i_2_c = calib_par->R_i_2_c ;
    Odometry->T_i_2_c = calib_par->T_i_2_c ;

    std::cout << "R_i_2_c:\n" << Odometry->R_i_2_c << "\n" ;
    std::cout << "T_i_2_c:\n" << Odometry->T_i_2_c.transpose() << "\n" ;


	imageSeqNumber = 0;
    sumDist = 0 ;
    cnt_info_smooth = 0 ;
    to_pub_info.x = 0 ;
    to_pub_info.y = 0 ;
    to_pub_info.z = 0 ;

    image0Buf.clear();
    image1Buf.clear();
    imuQueue.clear();
}


LiveSLAMWrapper::~LiveSLAMWrapper()
{
    if(Odometry != 0)
        delete Odometry;
    if( outFile.is_open() )
	{
        outFile.flush();
        outFile.close();
	}
    image0Buf.clear();
    image1Buf.clear();
    imuQueue.clear();
}

void LiveSLAMWrapper::popAndSetGravity( int skipFrameNum )
{
    unsigned int image0BufSize ;
    unsigned int image1BufSize ;
    std::list<ImageMeasurement>::reverse_iterator reverse_iterImage ;
    ros::Time tImage ;
    ros::Rate r(1000) ;

    gravity_b0.setZero() ;
    Vector3d acc_avg, gyr_avg ;
    acc_avg.setZero() ;
    gyr_avg.setZero() ;
    while ( nh.ok() )
    {
        ros::spinOnce() ;
        image0_queue_mtx.lock();
        image1_queue_mtx.lock();
        imu_queue_mtx.lock();
        image0BufSize = image0Buf.size();
        image1BufSize = image1Buf.size();
        if ( image0BufSize < 20 || image1BufSize < 20
              ){
            image0_queue_mtx.unlock();
            image1_queue_mtx.unlock();
            imu_queue_mtx.unlock();
            r.sleep() ;
            continue ;
        }
        reverse_iterImage = image0Buf.rbegin() ;
        tImage = reverse_iterImage->t ;
        reverse_iterImage = image1Buf.rbegin() ;
        if ( reverse_iterImage->t < tImage ){
            tImage = reverse_iterImage->t ;
        }
        pImage0Iter = image0Buf.begin();
        pImage1Iter = image1Buf.begin();
        while ( pImage0Iter->t < tImage ){
            pImage0Iter = image0Buf.erase( pImage0Iter ) ;
        }
        while ( pImage1Iter->t < tImage ){
            pImage1Iter = image1Buf.erase( pImage1Iter ) ;
        }
        image0_queue_mtx.unlock();
        image1_queue_mtx.unlock();

        cout << "tImage " << tImage << "\n" ;
        //imu_queue_mtx.lock();
        while ( imuQueue.size() < 1 || imuQueue.rbegin()->header.stamp < tImage ){
            //cout << "currentIMU_iter->header.stamp " << imuQueue.rbegin()->header.stamp  << "\n" ;
            imu_queue_mtx.unlock();
            r.sleep() ;
            ros::spinOnce() ;
            imu_queue_mtx.lock();
        }
        imu_queue_mtx.unlock();

        int imuNum = 0;
        currentIMU_iter = imuQueue.begin() ;
        while( currentIMU_iter->header.stamp < tImage )
        {
            //cout << "currentIMU_iter->header.stamp " << currentIMU_iter->header.stamp << "\n" ;
            imuNum++;
            gravity_b0(0) += currentIMU_iter->linear_acceleration.x;
            gravity_b0(1) += currentIMU_iter->linear_acceleration.y;
            gravity_b0(2) += currentIMU_iter->linear_acceleration.z;
            currentIMU_iter = imuQueue.erase(currentIMU_iter);

            acc_avg(0) += currentIMU_iter->linear_acceleration.x;
            acc_avg(1) += currentIMU_iter->linear_acceleration.y;
            acc_avg(2) += currentIMU_iter->linear_acceleration.z;

            gyr_avg(0) += currentIMU_iter->angular_velocity.x ;
            gyr_avg(1) += currentIMU_iter->angular_velocity.y ;
            gyr_avg(2) += currentIMU_iter->angular_velocity.z ;
        }
        imu_queue_mtx.unlock();
        gravity_b0 /= imuNum ;
        acc_avg /= imuNum ;
        gyr_avg /= imuNum ;
        //gravity_b0 = -gravity_b0 ;
        break ;
    }
    initialTime = tImage ;

    cout << "initial time " << initialTime << "\n" ;
    //1413393213480760000

    cout << "acc_avg = " <<  acc_avg.transpose() << "\n" ;
    cout << "gyr_avg = " << gyr_avg.transpose() << "\n" ;

    Odometry->acc_0 = acc_avg;
    Odometry->gyr_0 = gyr_avg;
    Odometry->initGravity(gravity_b0) ;
    //Odometry->eigen2para() ;

    //imu_queue_mtx.unlock();
    cv::Mat image0 = pImage0Iter->image.clone();
    cv::Mat image1 = pImage1Iter->image.clone();
    //image1_queue_mtx.unlock();
    //image0_queue_mtx.unlock();

    cv::Mat disparity, iDepth, iVar ;
    Odometry->bm_->compute(image0, image1, disparity);
    disparity.convertTo(disparity, CV_32F);
    disparity /= 16 ;

//    int SADWindowSize = 17 ;
//    int maxDisparity = 128 ;
//    sgbm_.preFilterCap = 63;
//    sgbm_.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;
//    sgbm_.P1 = 8*sgbm_.SADWindowSize*sgbm_.SADWindowSize;
//    sgbm_.P2 = 32*sgbm_.SADWindowSize*sgbm_.SADWindowSize;
//    sgbm_.minDisparity = 0;
//    sgbm_.numberOfDisparities = maxDisparity;
//    sgbm_.uniquenessRatio = 10;
//    sgbm_.speckleWindowSize = 100;
//    sgbm_.speckleRange = 32;
//    sgbm_.disp12MaxDiff = 1;
//    sgbm_.fullDP = false;
//    sgbm_.numberOfDisparities = maxDisparity ;
//    sgbm_.SADWindowSize = SADWindowSize ;
//    sgbm_(image0, image1, disparity ) ;
//    disparity.convertTo(disparity, CV_32F);
//    disparity /= 16.0 ;

    calculateInvDepthImage(disparity, iDepth, iVar, 0.11, fx );

    Odometry->pKF = Odometry->states[0] ;
    Odometry->lastKF = Odometry->pKF ;
    Odometry->pKF->reset(tImage, image0);
    Odometry->pKF->insertDepth( iDepth, iVar );
    Odometry->pKF->initEdgePyramid();
    Odometry->pKF->initDistanceTransformPyramid();
    Odometry->pKF->initPointsPyramid( cali );
    Odometry->pKF->keyFrameFlag = true ;
    Odometry->pKF->cameraLink.clear();
    Odometry->pKF->state_id = 0 ;
    Odometry->cR_64 = Eigen::Matrix3d::Identity() ;
    Odometry->cT_64 = Eigen::Vector3d::Zero() ;
    Odometry->frame_count = 0 ;

    //dense related
    Sophus::Matrix3f K_sophus;
    K_sophus << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    Odometry->pKF->frame.reset( new lsd_slam::Frame( 0, width, height, K_sophus, tImage.toSec(), image0.data) ) ;
    //puts("1") ;
    Odometry->currentKeyFrame = Odometry->pKF->frame ;
    //puts("2") ;
    Odometry->currentKeyFrame->setInvDepthFromGroundTruth( (float*)iDepth.data ) ;
    //puts("3") ;
    Odometry->RefToFrame = SE3();
    //puts("3.5") ;
    Odometry->updateTrackingReference();
    //puts("4") ;
    //Odometry->margin.initPrior();

    if ( printDebugInfo )
    {/*
        sensor_msgs::Image msg;
        msg.header.stamp = tImage;
        sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::MONO8, height,
                               width, width, Odometry->states[0]->edge[0].data );
        pub_gradientMapForDebug.publish(msg) ;*/
        if ( denseOrNot )
        {
             sensor_msgs::Image msg;
            cv::Mat gradientMapForDebug ;
            cv::cvtColor(image0, gradientMapForDebug, CV_GRAY2BGR);
            Odometry->generateDubugMap(Odometry->states[0], gradientMapForDebug);
            msg.header.stamp = tImage;
            sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::BGR8, height,
                                   width, width*3, gradientMapForDebug.data );
            pub_gradientMapForDebug.publish(msg) ;
        }
        else
        {
             sensor_msgs::Image msg;
            msg.header.stamp = tImage;
            sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::MONO8, height,
                                   width, width, Odometry->states[0]->edge[0].data );
            pub_gradientMapForDebug.publish(msg) ;
        }
    }




//    pubOdometry( Odometry->Ps[0], Odometry->Vs[0], Odometry->Rs[0],
//            pub_odometry, pub_pose, pub_nav_Odometry, 0, Eigen::Matrix3d::Identity(),
//            true, tImage );
    lastLoopClorsureTime = tImage ;
}

void LiveSLAMWrapper::pubCameraLink()
{
//    cv::Mat linkListMap(500, 500, CV_8UC3 ) ;
//    linkListMap.setTo( cv::Vec3b(200,200,200));
//    cv::Vector<cv::Point2f> locations(slidingWindowSize) ;
//    double angle_K = 2.0*PI/slidingWindowSize ;
//    double r = 200.0 ;
//    for ( int i = 0 ; i < slidingWindowSize ; i++ )
//    {
//        locations[i].x = sin(angle_K*i)*r + 250.0 ;
//        locations[i].y = cos(angle_K*i)*r + 250.0 ;
//        if ( Odometry->slidingWindow[i].isReady == false){
//            continue ;
//        }
//        if ( Odometry->slidingWindow[i].keyFrameFlag ){
//            cv::circle(linkListMap, locations[i], 6, cv::Scalar(255, 0, 0), 5);
//        }
//        else{
//            cv::circle(linkListMap, locations[i], 6, cv::Scalar(0, 0, 255), 5);
//        }
//        if ( i == Odometry->head ){
//            cv::circle(linkListMap, locations[i], 6, cv::Scalar(0, 255, 0), 5);
//        }
//        cv::putText(linkListMap, boost::to_string(i), locations[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 1);
//    }
//    int cnt = 0 ;
//    for( int i = 0 ; i < Odometry->numOfState ; i++ )
//    {
//        int idx = Odometry->head + i ;
//        if ( idx >= slidingWindowSize ){
//            idx -= slidingWindowSize ;
//        }
//        if ( Odometry->slidingWindow[i].isReady == false ){
//            continue ;
//        }
//        if ( Odometry->slidingWindow[idx].keyFrameFlag == false){
//            continue;
//        }
//        list<int>::iterator iter =  Odometry->slidingWindow[idx].cameraLinkList.begin();
//        for (; iter !=  Odometry->slidingWindow[idx].cameraLinkList.end(); iter++ )
//        {
//            int linkID = *iter;
//            cv::line(linkListMap, locations[idx], locations[linkID], cv::Scalar(0, 255, 255), 3);
//            cnt++ ;
//        }
//    }
//    cv::imshow("linkListMap", linkListMap ) ;
//    cv::waitKey(1) ;
}

//void LiveSLAMWrapper::pubPointCloud(int num, ros::Time imageTimeStamp, Eigen::Matrix3d R_vi_2_odometry )
//{
//    sensor_msgs::PointCloud2 pc2 ;
//    pc2.header.frame_id = "/map";//world
//    pc2.header.stamp = imageTimeStamp ;
//    pc2.height = 1 ;
//    pc2.width = num ;
//    pc2.is_bigendian = false ;
//    pc2.is_dense = true ;
//    pc2.point_step = sizeof(float) * 3 ;
//    pc2.row_step = pc2.point_step * pc2.width ;

//    sensor_msgs::PointField field;
//    pc2.fields.resize(3);
//    string f_name[3] = {"x", "y", "z"};
//    for (size_t idx = 0; idx < 3; ++idx)
//    {
//        field.name = f_name[idx];
//        field.offset = idx * sizeof(float);
//        field.datatype = sensor_msgs::PointField::FLOAT32;
//        field.count = 1;
//        pc2.fields[idx] = field;
//    }
//    pc2.data.clear();
//    pc2.data.reserve( pc2.row_step );

//    vector<float> pt32;
//    pt32.resize(num*3);
//    int level = 0 ;
//    int w = Odometry->currentKeyFrame->width(level);
//    int h = Odometry->currentKeyFrame->height(level);
//    float fxInvLevel = Odometry->currentKeyFrame->fxInv(level);
//    float fyInvLevel = Odometry->currentKeyFrame->fxInv(level);
//    float cxInvLevel = Odometry->currentKeyFrame->fxInv(level);
//    float cyInvLevel = Odometry->currentKeyFrame->fxInv(level);
//    const float* pyrIdepthSource = Odometry->currentKeyFrame->idepth(level);
//    const float* pyrIdepthVarSource = Odometry->currentKeyFrame->idepthVar(level);
//    Eigen::Vector3f posDataPT ;
//    Eigen::Vector3f posDataOutput ;
//    Eigen::Matrix3f R_output ;
//    R_output << R_vi_2_odometry(0, 0), R_vi_2_odometry(0, 1), R_vi_2_odometry(0, 2),
//            R_vi_2_odometry(1, 0), R_vi_2_odometry(1, 1), R_vi_2_odometry(1, 2),
//            R_vi_2_odometry(2, 0), R_vi_2_odometry(2, 1), R_vi_2_odometry(2, 2) ;

//    int k = 0 ;
//    for(int x=1; x<w-1; x++)
//    {
//        for(int y=1; y<h-1; y++)
//        {
//            int idx = x + y*w;

//            if(pyrIdepthVarSource[idx] <= 0 || pyrIdepthSource[idx] == 0) continue;

//            posDataPT = (1.0f / pyrIdepthSource[idx]) * Eigen::Vector3f(fxInvLevel*x+cxInvLevel,fyInvLevel*y+cyInvLevel,1);
//            posDataOutput = R_output*posDataPT ;
//            pt32[k++] = posDataOutput(0) ;
//            pt32[k++] = posDataOutput(1) ;
//            pt32[k++] = posDataOutput(2) ;
//        }
//    }

//    uchar * pt_int = reinterpret_cast<uchar *>(pt32.data());
//    for (size_t idx = 0; idx < pc2.row_step; ++idx){
//        pc2.data.push_back(pt_int[idx]);
//    }
//    Odometry->pub_cloud.publish(pc2) ;
//}



void LiveSLAMWrapper::BALoop()
{
    ros::Rate BARate(100) ;
    ros::Rate trackingOnSleep(100) ;
    list<ImageMeasurement>::iterator iterImage ;
    std::list<visensor_node::visensor_imu>::iterator iterIMU ;
    cv::Mat image0 ;
    cv::Mat image1 ;
    cv::Mat gradientMapForDebug(height, width, CV_8UC3) ;
    sensor_msgs::Image msg;
    double t ;

    while ( nh.ok() )
    {
        Odometry->frameInfoList_mtx.lock();
        int ttt = (Odometry->frameInfoListTail - Odometry->frameInfoListHead);
        if ( ttt < 0 ){
            ttt += frameInfoListSize ;
        }
        printf("[BA thread] sz=%d\n", ttt ) ;
        if ( ttt < 1 ){
            Odometry->frameInfoList_mtx.unlock();
            BARate.sleep() ;
            continue ;
        }
        for ( int sz ; ; )
        {
            Odometry->frameInfoListHead++ ;
            if ( Odometry->frameInfoListHead >= frameInfoListSize ){
                Odometry->frameInfoListHead -= frameInfoListSize ;
            }
            sz = Odometry->frameInfoListTail - Odometry->frameInfoListHead ;
            if ( sz == 0 ){
                break ;
            }
            if ( Odometry->frameInfoList[Odometry->frameInfoListHead].keyFrameFlag ){
                break ;
            }
        }
//        if ( Odometry->frameInfoList[Odometry->frameInfoListHead].keyFrameFlag ){
//            puts("k") ;
//        }
//        else{
//            puts("c") ;
//        }
        ros::Time imageTimeStamp = Odometry->frameInfoList[Odometry->frameInfoListHead].t ;
        Odometry->frameInfoList_mtx.unlock();

        //Pop out the image list
        image1_queue_mtx.lock();
        iterImage = image1Buf.begin() ;
        while ( iterImage->t < imageTimeStamp ){
            iterImage = image1Buf.erase( iterImage ) ;
        }
        image1 = iterImage->image.clone();
        image1_queue_mtx.unlock();

        image0_queue_mtx.lock();
        iterImage = image0Buf.begin() ;
        while ( iterImage->t < imageTimeStamp ){
            iterImage = image0Buf.erase( iterImage ) ;
        }
        image0 = iterImage->image.clone();
        image0_queue_mtx.unlock();

        imu_queue_mtx.lock();
        iterIMU = imuQueue.begin() ;
        Vector3d linear_acceleration;
        Vector3d angular_velocity;

        Odometry->frame_count++ ;
        //std::cout << "imageTime=" << imageTimeStamp << std::endl;
        while ( iterIMU->header.stamp < imageTimeStamp )
        {
            linear_acceleration(0) = iterIMU->linear_acceleration.x;
            linear_acceleration(1) = iterIMU->linear_acceleration.y;
            linear_acceleration(2) = iterIMU->linear_acceleration.z;
            angular_velocity(0) = iterIMU->angular_velocity.x;
            angular_velocity(1) = iterIMU->angular_velocity.y;
            angular_velocity(2) = iterIMU->angular_velocity.z;

//            to_pub_info.x = angular_velocity(0)*180/PI ;
//            to_pub_info.y = angular_velocity(1)*180/PI ;
//            to_pub_info.z = angular_velocity(2)*180/PI ;
//            Odometry->pub_angular_velocity.publish( to_pub_info ) ;
//            outFile << to_pub_info.x << " "
//                    << to_pub_info.y << " "
//                    << to_pub_info.z << "\n";

            double pre_t = iterIMU->header.stamp.toSec();
            iterIMU = imuQueue.erase(iterIMU);
            double next_t = iterIMU->header.stamp.toSec();
            double dt = next_t - pre_t ;

//            std::cout << linear_acceleration.transpose() << std::endl ;
//            std::cout << angular_velocity.transpose() << std::endl ;
            Odometry->processIMU( dt, linear_acceleration, angular_velocity );
        }
        imu_queue_mtx.unlock();

        Odometry->insertFrame(imageTimeStamp, image0);
        STATE* nowFrame = Odometry->states[Odometry->frame_count] ;

        if ( printDebugInfo )
        {
            msg.header.stamp = imageTimeStamp;
            sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::MONO8, height,
                                   width, width, nowFrame->edge[0].data );
            pub_edge.publish(msg) ;
            cv::Mat showDT ;
            cv::normalize(nowFrame->distanceTransformMap[0], showDT, 0, 255, CV_MINMAX, CV_8U);

            sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::MONO8, height,
                                   width, width, showDT.data );
            pub_distanceTransform.publish(msg) ;
        }

        bool noUse = false ;
        {//consistent check
            unique_lock<std::mutex> lock(Odometry->KF_mutex);

            Vector3d tmp = Utility::R2ypr( Odometry->Rs[Odometry->frame_count].transpose()*Odometry->Rs[Odometry->lastKF->state_id]
                    *Odometry->frameInfoList[Odometry->frameInfoListHead].R_k_2_c.transpose() ) ;
            double errorAngle = tmp.norm() ;
            Vector3d errorT = Odometry->Rs[Odometry->frame_count].transpose()
                    *(Odometry->Ps[Odometry->lastKF->state_id] - Odometry->Ps[Odometry->frame_count])
                    - Odometry->frameInfoList[Odometry->frameInfoListHead].T_k_2_c ;
            double errorTranslation = errorT.norm();

//            std::cout << "Rotation-IMU Prior: " << Odometry->Rs[Odometry->frame_count].transpose()*Odometry->Rs[Odometry->lastKF->state_id] << endl;
//            std::cout << "Visual: " << Odometry->frameInfoList[Odometry->frameInfoListHead].R_k_2_c.transpose() << endl;
//            std::cout << "diff: " << errorQ.vec().transpose() << endl ;

//            std::cout << "Translation-IMU Prior: " << (Odometry->Rs[Odometry->frame_count].transpose()
//                    *(Odometry->Ps[Odometry->lastKF->state_id] - Odometry->Ps[Odometry->frame_count])).transpose() << endl;
//            std::cout << "Visual: " << Odometry->frameInfoList[Odometry->frameInfoListHead].T_k_2_c.transpose() << endl;
//            std::cout << "diff: " << errorT.transpose() << endl ;

            if ( IMUorNot == false ){
                cout << "Angle: " << Utility::R2ypr(Odometry->frameInfoList[Odometry->frameInfoListHead].R_k_2_c).transpose() << endl;
                cout << "Translation: " << Odometry->frameInfoList[Odometry->frameInfoListHead].T_k_2_c.transpose() << endl;
            }

            //
            if ( IMUorNot && errorAngle > errorAngleThreshold ){
                ROS_WARN("errorAngle=%f errorAngleThreshold=%f", errorAngle, errorAngleThreshold ) ;
                cout << "IMU Prior: " << Utility::R2ypr(Odometry->Rs[Odometry->frame_count].transpose()*Odometry->Rs[Odometry->lastKF->state_id]).transpose() << endl;
                cout << "Visual: " << Utility::R2ypr(Odometry->frameInfoList[Odometry->frameInfoListHead].R_k_2_c).transpose() << endl;
                cout << "diff: " << tmp.transpose() << endl ;
                noUse = true ;
            }
            if ( IMUorNot && errorTranslation > errorTranslationThreshold ){
                ROS_WARN("errorTranslation=%f errorTranslationThreshold=%f", errorTranslation, errorTranslationThreshold ) ;
                cout << "IMU Prior: " << (Odometry->Rs[Odometry->frame_count].transpose()
                        *(Odometry->Ps[Odometry->lastKF->state_id] - Odometry->Ps[Odometry->frame_count])).transpose() << endl;
                cout << "Visual: " << Odometry->frameInfoList[Odometry->frameInfoListHead].T_k_2_c.transpose() << endl;
                cout << "diff: " << errorT.transpose() << endl ;
                noUse = true ;
            }
        }
        //noUse = false ;

        if ( noUse )
        {
            Odometry->frameInfoList[Odometry->frameInfoListHead].keyFrameFlag = true ;
            Odometry->frameInfoList[Odometry->frameInfoListHead].trust = false ;

            Odometry->tracking_mtx.lock();
            Odometry->lock_densetracking = true;
            Odometry->tracking_mtx.unlock();

            //Odometry->onTracking_mtx.lock();
            while ( Odometry->onTracking )
            {
                //Odometry->onTracking_mtx.unlock();
                trackingOnSleep.sleep();
                //puts("BA-sleep") ;
                //Odometry->onTracking_mtx.lock();
            }
            //puts("endofsleep") ;
            //Odometry->onTracking_mtx.unlock();
        }
        if ( Odometry->frameInfoList[Odometry->frameInfoListHead].keyFrameFlag )
        {

            //update the reference frame
            Odometry->pKF = Odometry->states[Odometry->frame_count] ;
            cv::Mat disparity, iDepth, iVar ;

            Odometry->bm_->compute(image0, image1, disparity);
            disparity.convertTo(disparity, CV_32F);
            disparity /= 16 ;

//            cv::imshow("image0", image0 ) ;
//            cv::imshow("image1", image1 ) ;
//            cv::imshow("disparity", disparity/64 ) ;
//            cv::waitKey(3) ;

//            int SADWindowSize = 17 ;
//            int maxDisparity = 128 ;
//            sgbm_.preFilterCap = 63;
//            sgbm_.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;
//            sgbm_.P1 = 8*sgbm_.SADWindowSize*sgbm_.SADWindowSize;
//            sgbm_.P2 = 32*sgbm_.SADWindowSize*sgbm_.SADWindowSize;
//            sgbm_.minDisparity = 0;
//            sgbm_.numberOfDisparities = maxDisparity;
//            sgbm_.uniquenessRatio = 10;
//            sgbm_.speckleWindowSize = 100;
//            sgbm_.speckleRange = 32;
//            sgbm_.disp12MaxDiff = 1;
//            sgbm_.fullDP = false;
//            sgbm_.numberOfDisparities = maxDisparity ;
//            sgbm_.SADWindowSize = SADWindowSize ;
//            sgbm_(image0, image1, disparity ) ;
//            disparity.convertTo(disparity, CV_32F );
//            disparity /= 16.0 ;

            calculateInvDepthImage(disparity, iDepth, iVar, 0.11, fx);
            nowFrame->insertDepth(iDepth, iVar);
            nowFrame->initPointsPyramid(cali);

#ifdef DENSE
            nowFrame->frame->setInvDepthFromGroundTruth( (float*)iDepth.data) ;
            Odometry->currentKeyFrame = Odometry->pKF->frame ;

            //reset the initial guess
            Odometry->RefToFrame = SE3() ;
            //update tracking reference
            Odometry->updateTrackingReference();
#endif

            if ( printDebugInfo )
            {
                if ( denseOrNot )
                {
                    cv::cvtColor(image0, gradientMapForDebug, CV_GRAY2BGR);
                    Odometry->generateDubugMap(nowFrame, gradientMapForDebug);
                    msg.header.stamp = imageTimeStamp;
                    sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::BGR8, height,
                                           width, width*3, gradientMapForDebug.data );
                    pub_gradientMapForDebug.publish(msg) ;
                }
                else
                {
                    msg.header.stamp = imageTimeStamp;
                    sensor_msgs::fillImage(msg, sensor_msgs::image_encodings::MONO8, height,
                                           width, width, nowFrame->edge[0].data );
                    pub_gradientMapForDebug.publish(msg) ;
                }

            }

            nowFrame->keyFrameFlag = true ;
            nowFrame->cameraLink.clear();
            //Odometry->currentKeyFrame.initPointsPyramid( cali );
            //Odometry->currentKeyFrame.keyFrameFlag = true ;
            //Odometry->currentKeyFrame.cameraLinkList.clear() ;
            Odometry->cR_64 = Eigen::Matrix3d::Identity() ;
            Odometry->cT_64 = Eigen::Vector3d::Zero() ;

            //unlock dense tracking
            Odometry->tracking_mtx.lock();
            Odometry->lock_densetracking = false;
            Odometry->tracking_mtx.unlock();

            if ( enable_LoopClosure
                  && (imageTimeStamp - lastLoopClorsureTime).toSec() > loopClosureInterval
                 )
            {
                //add possible loop closure link
                t = (double)cvGetTickCount();
                ROS_INFO("before loop closure") ;
                if ( denseOrNot ){
                    Odometry->setReprojectionListRelateToLastestKeyFrameDense() ;
                }
                else{
                    Odometry->setReprojectionListRelateToLastestKeyFrameEdgeCeres() ;
                }
                ROS_WARN("loop closure link cost time: %f", ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000) );
                t = (double)cvGetTickCount()  ;
                lastLoopClorsureTime = imageTimeStamp ;
            }
        }



        if ( Odometry->frameInfoList[Odometry->frameInfoListHead].trust )
        {
//            Eigen::Matrix3d r ;
//            r.setIdentity();
//            Eigen::Vector3d t ;
//            t.setZero() ;
//            Odometry->insertCameraLink( Odometry->lastKF, nowFrame,
//                          r,
//                          t,
//                          Odometry->frameInfoList[Odometry->frameInfoListHead].lastestATA );
            Odometry->insertCameraLink( Odometry->lastKF, nowFrame,
                          Odometry->frameInfoList[Odometry->frameInfoListHead].R_k_2_c,
                          Odometry->frameInfoList[Odometry->frameInfoListHead].T_k_2_c,
                          Odometry->frameInfoList[Odometry->frameInfoListHead].lastestATA );

            //Vector3d tmp = Utility::R2ypr(Odometry->frameInfoList[Odometry->frameInfoListHead].R_k_2_c) ;
            //ROS_WARN("[insert Frame R] %lf %lf %lf", tmp(0), tmp(1), tmp(2) ) ;

//            cout << "T " << -Odometry->frameInfoList[Odometry->frameInfoListHead].T_k_2_c.transpose() << endl ;
//            Quaterniond dq(Odometry->frameInfoList[Odometry->frameInfoListHead].R_k_2_c) ;
//            cout << "R " << dq.x() << " " << dq.y() << " " << dq.z() << endl ;
        }

        int control_flag = 0 ;
        Vector3d preBAt = Odometry->Ps[Odometry->frame_count] ;

//        cout << "[-BA]current Position: " << nowFrame->T_bk_2_b0.transpose() << endl;
//        cout << "[-BA]current Velocity: " << nowFrame->v_bk.transpose() << endl;

        //BA
        t = (double)cvGetTickCount()  ;

        //Odometry->BA();

        Odometry->solve_ceres();

        printf("BA cost time: %f\n", ((double)cvGetTickCount() - t) / (cvGetTickFrequency() * 1000) );

        Odometry->lastKF = Odometry->pKF ;
        ROS_INFO("lastKF ID = %d", Odometry->lastKF->state_id ) ;

        cout << "[BA-]current Position: " << Odometry->Ps[Odometry->frame_count].transpose() << endl;
        cout << "[BA-]current Velocity: " << Odometry->Vs[Odometry->frame_count].transpose() << endl;
        cout << "[BA-]current bias_acc: " << Odometry->Bas[Odometry->frame_count].transpose() << endl;
        cout << "[BA-]current bias_gyr: " << Odometry->Bgs[Odometry->frame_count].transpose() << endl;

        //printf("before marginalization, tail = %d\n", Odometry->tail ) ;

        if ( (Odometry->Ps[Odometry->frame_count] - preBAt ).norm() > 0.1 ){
            control_flag = 1 ; //loop_closure or other sudden position change case
        }
        if ( Odometry->frameInfoList[Odometry->frameInfoListHead].trust == false ){
            control_flag = 2 ; //only IMU link, dense tracking fails
        }

        if ( visaulizeGraphStructure ){
            pubCameraLink();
        }

        //marginalziation
        //Odometry->twoWayMarginalize();
        //Odometry->setNewMarginalzationFlag();

        //printf("after marginalization, tail = %d\n", Odometry->tail ) ;

  //      if ( Odometry->frame_count >= WINDOW_SIZE-1 )
        {
            if ( onUAV ){
                pubOdometry(Odometry->Ps[Odometry->frame_count],
                        Odometry->Vs[Odometry->frame_count],
                        Odometry->Rs[Odometry->frame_count],
                        pub_odometry, pub_pose, pub_nav_Odometry,
                        control_flag, R_vi_2_odometry,
                        Odometry->frameInfoList[Odometry->frameInfoListHead].keyFrameFlag, imageTimeStamp );
            }
            else{
                pubOdometry(Odometry->Ps[Odometry->frame_count],
                        Odometry->Vs[Odometry->frame_count],
                        Odometry->Rs[Odometry->frame_count],
                        pub_odometry, pub_pose, pub_nav_Odometry,
                        control_flag, Eigen::Matrix3d::Identity(),
                        Odometry->frameInfoList[Odometry->frameInfoListHead].keyFrameFlag, imageTimeStamp );
            }
        }

        if ( printDebugInfo )
        {
            int colorFlag = 0 ;

            if ( Odometry->frameInfoList[Odometry->frameInfoListHead].keyFrameFlag == false ){
                colorFlag = 0 ;
            }
            if ( noUse ){
                colorFlag = 2 ;
            }
            if ( Odometry->frameInfoList[Odometry->frameInfoListHead].keyFrameFlag && noUse == false ){
                colorFlag = 1 ;
            }
//            colorFlag = Odometry->frameInfoList[Odometry->frameInfoListHead].keyFrameFlag ;
//            if (  Odometry->frameInfoList[Odometry->frameInfoListHead].trust == false ){
//                colorFlag = 2 ;
//            }
            outFile << colorFlag << " " << Odometry->Ps[Odometry->frame_count](0) << " "
                        << Odometry->Ps[Odometry->frame_count](1) << " "
                        << Odometry->Ps[Odometry->frame_count](2) << "\n" ;
            outFile.flush();

            if ( onUAV ){
                pubPath(Odometry->Ps[Odometry->frame_count],
                        colorFlag,
                        path_line, pub_path, R_vi_2_odometry );
            }
            else{
                pubPath(Odometry->Ps[Odometry->frame_count],
                        colorFlag,
                        path_line, pub_path, Eigen::Matrix3d::Identity() );
            }
        }

//        ros::Time tt ;
//        tt.sec = 1 ;
//        tt.nsec = 2 ;

        if ( enable_pubTF )
        {
            static tf::TransformBroadcaster br;
            tf::Transform transform;
            //Vector3d t_translation = R_vi_2_odometry * Odometry->Ps[Odometry->frame_count] ;
            Vector3d t_translation = Odometry->Ps[Odometry->frame_count] ;
            transform.setOrigin(tf::Vector3(t_translation(0),
                                            t_translation(1),
                                            t_translation(2)) );
            Quaterniond t_q(Odometry->Rs[Odometry->frame_count]) ;
            tf::Quaternion q;
            //Quaterniond tt_q(R_vi_2_odometry * Odometry->Rs[Odometry->frame_count] * R_vi_2_odometry.transpose());
            q.setW(t_q.w());
            q.setX(t_q.x());
            q.setY(t_q.y());
            q.setZ(t_q.z());
            transform.setRotation(q);
            br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "densebody"));
        }

//        if ( noUse ){
//            Odometry->frameInfoList_mtx.lock();
//            int ttt = (Odometry->frameInfoListTail - Odometry->frameInfoListHead);
//            if ( ttt < 0 ){
//                ttt += frameInfoListSize ;
//            }
//            if ( ttt >= 1 )
//            {
//                Odometry->frameInfoListHead += ttt ;
//                if ( Odometry->frameInfoListHead >= frameInfoListSize ){
//                    Odometry->frameInfoListHead -= frameInfoListSize ;
//                }
//            }
//            Odometry->frameInfoList_mtx.unlock();
//        }

//        int preIndex = Odometry->tail - 1 ;
//        if ( preIndex < 0 ){
//            preIndex += slidingWindowSize ;
//        }
//        Vector3d tt_dist = (Odometry->slidingWindow[Odometry->tail]->T_bk_2_b0 -
//                Odometry->slidingWindow[preIndex]->T_bk_2_b0) ;
//        Matrix3d tt_rotate = Odometry->slidingWindow[Odometry->tail]->R_bk_2_b0.transpose() *
//                Odometry->slidingWindow[preIndex]->R_bk_2_b0 ;
//        Quaterniond tt_q(tt_rotate) ;

//        to_pub_info.x = Odometry->slidingWindow[Odometry->tail]->v_bk(0) ;
//        to_pub_info.y = Odometry->slidingWindow[Odometry->tail]->v_bk(1) ;
//        to_pub_info.z = Odometry->slidingWindow[Odometry->tail]->v_bk(2) ;
//        Odometry->pub_linear_velocity.publish(to_pub_info) ;


    }
}

void LiveSLAMWrapper::Loop()
{
    std::list<visensor_node::visensor_imu>::reverse_iterator reverse_iterImu ;
    std::list<ImageMeasurement>::iterator  pIter ;
    ros::Time imageTimeStamp ;
    cv::Mat   image0 ;
    cv::Mat   image1 ;
    ros::Rate r(100.0);
    while ( nh.ok() )
    {
        Odometry->tracking_mtx.lock();
        bool tmpFlag = Odometry->lock_densetracking ;
        Odometry->tracking_mtx.unlock();
        if ( tmpFlag == true ){
            r.sleep() ;
            continue ;
        }

        image0_queue_mtx.lock();
        image1_queue_mtx.lock();
        imu_queue_mtx.lock();
        pIter = pImage0Iter ;
        pIter++ ;
        if ( pIter == image0Buf.end() ){
            image0_queue_mtx.unlock();
            image1_queue_mtx.unlock();
            imu_queue_mtx.unlock();
            r.sleep() ;
            continue ;
        }
        imageTimeStamp = pIter->t ;

        pIter = pImage1Iter ;
        pIter++ ;
        if ( pIter == image1Buf.end() ){
            image0_queue_mtx.unlock();
            image1_queue_mtx.unlock();
            imu_queue_mtx.unlock();
            r.sleep() ;
            continue ;
        }
        if ( image1Buf.rbegin()->t < imageTimeStamp )
        {
            image0_queue_mtx.unlock();
            image1_queue_mtx.unlock();
            imu_queue_mtx.unlock();
            r.sleep() ;
            continue ;
        }
        reverse_iterImu = imuQueue.rbegin() ;
//        printf("%d %d\n", imuQueue.size() < 10, reverse_iterImu->header.stamp <= imageTimeStamp ) ;
        if ( imuQueue.size() < 1 || reverse_iterImu->header.stamp < imageTimeStamp ){
            image0_queue_mtx.unlock();
            image1_queue_mtx.unlock();
            imu_queue_mtx.unlock();
            r.sleep() ;
            continue ;
        }
        imu_queue_mtx.unlock();

        while ( pImage1Iter->t < imageTimeStamp ){
            pImage1Iter++ ;
        }
        pImage0Iter++ ;
        //std::cout << imageTimeStamp.toNSec() << "\n" ;
        //std::cout << "[dt-image] " << imageTimeStamp << std::endl ;
        //std::cout << "[dt-imu] " << reverse_iterImu->header.stamp << " " << imuQueue.size() << std::endl ;
        ros::Time preTime = imageTimeStamp ;
        //pImage1Iter++ ;

        image1 = pImage1Iter->image.clone();
        image0 = pImage0Iter->image.clone();
        image1_queue_mtx.unlock();
        image0_queue_mtx.unlock();

        imu_queue_mtx.lock();
        Quaterniond q, dq ;
        q.setIdentity() ;
        Vector3d gyro_bias = Odometry->Bgs[Odometry->frame_count] ;
        while ( currentIMU_iter->header.stamp < imageTimeStamp )
        {
            double pre_t = currentIMU_iter->header.stamp.toSec();
            currentIMU_iter++ ;
            double next_t = currentIMU_iter->header.stamp.toSec();
            double dt = next_t - pre_t ;

            //prediction for dense tracking

            dq.x() = (currentIMU_iter->angular_velocity.x - gyro_bias(0) )*dt*0.5 ;
            dq.y() = (currentIMU_iter->angular_velocity.y - gyro_bias(1) )*dt*0.5 ;
            dq.z() = (currentIMU_iter->angular_velocity.z - gyro_bias(2) )*dt*0.5 ;

//            dq.x() = (currentIMU_iter->angular_velocity.x )*dt*0.5 ;
//            dq.y() = (currentIMU_iter->angular_velocity.y )*dt*0.5 ;
//            dq.z() = (currentIMU_iter->angular_velocity.z )*dt*0.5 ;
            dq.w() =  sqrt( 1 - SQ(dq.x()) * SQ(dq.y()) * SQ(dq.z()) ) ;
            if ( IMUorNot ){
                q = (q * dq).normalized();
            }
        }
        imu_queue_mtx.unlock();

		// process image
		//Util::displayImage("MyVideo", image.data);
        Matrix3d deltaR(q) ;
        deltaR = Odometry->R_i_2_c*deltaR*Odometry->R_i_2_c.transpose() ;

        ++imageSeqNumber;
        assert(image0.elemSize() == 1);
        assert(image1.elemSize() == 1);
        assert(fx != 0 || fy != 0);

        Odometry->onTracking_mtx.lock();
        Odometry->onTracking = true ;
        Odometry->onTracking_mtx.unlock();

        //puts("begin Tracking") ;

        struct timeval tv_start, tv_end;
        gettimeofday(&tv_start, NULL);
        if ( denseOrNot ){
            Odometry->trackFrameDense(image0, imageSeqNumber, imageTimeStamp, deltaR );
        }
        else{
            Odometry->trackFrameEdge(image0, image1, imageSeqNumber, imageTimeStamp, deltaR);
        }

        gettimeofday(&tv_end, NULL);
        float msTrackFrame = (tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f ;
        printf("msTrackFrame = %0.f\n", msTrackFrame ) ;

        //puts("end Tracking") ;

        Odometry->onTracking_mtx.lock();
        Odometry->onTracking = false ;
        Odometry->onTracking_mtx.unlock();
	}
}

void LiveSLAMWrapper::initRosPub()
{

    pub_path = nh.advertise<visualization_msgs::Marker>("/denseVO/path", 1000);
    pub_cloud = nh.advertise<sensor_msgs::PointCloud2>("/denseVO/cloud", 1000);
    pub_odometry = nh.advertise<quadrotor_msgs::Odometry>("/denseVO/odometry", 1000);
    pub_pose = nh.advertise<geometry_msgs::PoseStamped>("/denseVO/pose", 1000);
    pub_resudualMap = nh.advertise<sensor_msgs::Image>("denseVO/residualMap", 100 );
    pub_reprojectMap = nh.advertise<sensor_msgs::Image>("denseVO/reprojectMap", 100 );
    pub_edge = nh.advertise<sensor_msgs::Image>("denseVO/edge", 100 );;
    pub_distanceTransform = nh.advertise<sensor_msgs::Image>("denseVO/distanceTransform", 100 );;
    pub_gradientMapForDebug = nh.advertise<sensor_msgs::Image>("denseVO/debugMap", 100 );
    pub_denseTracking = nh.advertise<geometry_msgs::Vector3>("denseVO/dt", 100);
    pub_angular_velocity = nh.advertise<geometry_msgs::Vector3>("denseVO/angular_velocity", 100);
    pub_linear_velocity = nh.advertise<geometry_msgs::Vector3>("denseVO/linear_velocity", 100);
    pub_nav_Odometry = nh.advertise<nav_msgs::Odometry>("/denseVO/navOdometry", 100) ;

    path_line.header.frame_id    = "world";
    path_line.header.stamp       = ros::Time::now();
    path_line.ns                 = "edge_imu_bias";
    path_line.action             = visualization_msgs::Marker::ADD;
    path_line.pose.orientation.w = 1.0;
    path_line.type               = visualization_msgs::Marker::LINE_STRIP;
    path_line.scale.x            = 0.01 ;
    path_line.color.a            = 1.0;
    path_line.color.r            = 1.0;
    path_line.id                 = 1;
    path_line.points.push_back( geometry_msgs::Point());
    path_line.colors.push_back( std_msgs::ColorRGBA() );
    pub_path.publish(path_line);
}
