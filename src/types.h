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
#include "Eigen/Core"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"
#include "ros/ros.h"
#include "myMath.h"
#include "settings.h"
#include "Tracking/Frame.h"

struct CALIBRATION_PAR
{
    float fx[PYRAMID_LEVELS], fy[PYRAMID_LEVELS], cx[PYRAMID_LEVELS], cy[PYRAMID_LEVELS] ;
    float d[6] ;
    int width[PYRAMID_LEVELS] ;
    int height[PYRAMID_LEVELS] ;
    Eigen::Matrix3d R_i_2_c ;
    Eigen::Vector3d T_i_2_c ;
} ;

class ImageMeasurement
{
public:
    ros::Time t;
    cv::Mat   image;

    ImageMeasurement(const ros::Time& _t, const cv::Mat& _image)
    {
        t     = _t;
        image = _image.clone();
    }

    ImageMeasurement(const ImageMeasurement& i)
    {
        t     = i.t;
        image = i.image.clone();
    }

    ~ImageMeasurement() { }
};

struct FRAMEINFO
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix3d R_k_2_c;//R_k^(k+1)
    Eigen::Vector3d T_k_2_c;//T_k^(k+1)
    Matrix<double, 6, 6> lastestATA ;
    //Eigen::MatrixXd lastestATA ;
    bool keyFrameFlag ;
    bool trust ;
    ros::Time t ;
};

class CAMERALINK;

class STATE
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ros::Time tImage ;
    cv::Mat image[PYRAMID_LEVELS] ;
    cv::Mat idepth[PYRAMID_LEVELS] ;
    cv::Mat iVar[PYRAMID_LEVELS] ;
    cv::Mat edge[PYRAMID_LEVELS] ;
    int edgeNum[PYRAMID_LEVELS] ;
    cv::Mat distanceTransformMap[PYRAMID_LEVELS] ;

    Eigen::MatrixXf distanceTransformEigen[PYRAMID_LEVELS] ;
    Eigen::MatrixXf distanceTransformGradientX[PYRAMID_LEVELS] ;
    Eigen::MatrixXf distanceTransformGradientY[PYRAMID_LEVELS] ;
    Eigen::MatrixXf edge_3d[PYRAMID_LEVELS] ;
    Eigen::MatrixXf edge_2d[PYRAMID_LEVELS] ;

    bool makePointAndBuffer ;
    Eigen::Vector3f* posData[PYRAMID_LEVELS];
    float* posVar[PYRAMID_LEVELS] ;
    int numOfposData[PYRAMID_LEVELS] ;

    bool keyFrameFlag;
    bool isReady ;

    std::list<CAMERALINK> cameraLink;
    int state_id ;

    double iDepthMean ;
    std::shared_ptr<lsd_slam::Frame> frame ;

    int id(){
        return state_id;
    }

    void initPyramid()
    {
        int width = image[0].cols ;
        int height = image[0].rows ;
        for( int i = 1 ; i < optimizedPyramidLevel ; i++ )
        {
            width /= 2 ;
            height /= 2 ;

            image[i].create(height, width, CV_8U);
            for(int y=0; y<height; y++)
            {
                for(int x=0; x<width; x++ )
                {
                    //intensity
                    image[i].at<uchar>(y, x) =
                            ( (int)image[i-1].at<uchar>( y<<1, x<<1 ) +
                            (int)image[i-1].at<uchar>( y<<1, (x<<1)+1 ) +
                            (int)image[i-1].at<uchar>( (y<<1)+1, x<<1 ) +
                            (int)image[i-1].at<uchar>( (y<<1)+1, (x<<1)+1 ) ) / 4 + 0.5 ;
                }
            }

        }
    }

    void initDepthPyramid()
    {
        int width = image[0].cols ;
        int height = image[0].rows ;
        for( int i = 1 ; i < optimizedPyramidLevel ; i++ )
        {
            width /= 2 ;
            height /= 2 ;
            iVar[i].create(height, width, CV_32F) ;
            idepth[i].create(height, width, CV_32F) ;
            //            image[i] = cv::Mat::zeros(height, width, CV_8U ) ;
            for(int y=0; y<height; y++)
            {
                for(int x=0; x<width; x++ )
                {
                    //depth and variance
                    float idepthSumsSum = 0;
                    float ivarSumsSum = 0;
                    int num=0;

                    // build sums
                    float ivar;
                    float var = iVar[i-1].at<float>( y<<1, x<<1 ) ;
                    if(var > 0)
                    {
                        ivar = 1.0f / var;
                        ivarSumsSum += ivar;
                        idepthSumsSum += ivar * idepth[i-1].at<float>( y<<1, x<<1 );
                        num++;
                    }

                    var = iVar[i-1].at<float>( y<<1, (x<<1)+1 )  ;
                    if(var > 0)
                    {
                        ivar = 1.0f / var;
                        ivarSumsSum += ivar;
                        idepthSumsSum += ivar * idepth[i-1].at<float>( y<<1, (x<<1)+1 );
                        num++;
                    }

                    var = iVar[i-1].at<float>( (y<<1)+1, x<<1 )  ;
                    if(var > 0)
                    {
                        ivar = 1.0f / var;
                        ivarSumsSum += ivar;
                        idepthSumsSum += ivar * idepth[i-1].at<float>( (y<<1)+1, x<<1 )  ;
                        num++;
                    }

                    var =  iVar[i-1].at<float>( (y<<1)+1, (x<<1)+1 )  ;
                    if(var > 0)
                    {
                        ivar = 1.0f / var;
                        ivarSumsSum += ivar;
                        idepthSumsSum += ivar * idepth[i-1].at<float>( (y<<1)+1, (x<<1)+1 );
                        num++;
                    }

                    if(num > 0)
                    {
                        float depth = ivarSumsSum / idepthSumsSum;
                        idepth[i].at<float>(y, x) = 1.0f / depth;
                        iVar[i].at<float>(y, x) = num / ivarSumsSum;
                    }
                    else
                    {
                        idepth[i].at<float>(y, x) = -1.0;
                        iVar[i].at<float>(y, x) = -1.0;
                    }
                }
            }

        }
    }

    void initPointsPyramid(CALIBRATION_PAR* cali)
    {
        if ( makePointAndBuffer ){
            return ;
        }
        makePointAndBuffer = true ;
        int w, h ;
        for( int i = 0 ; i < optimizedPyramidLevel ; i++ )
        {
            w = edge[i].cols ;
            h = edge[i].rows ;

            //count the number of selected pixels
            int cnt = 0 ;
            for( int y = 0 ; y < h ; y++ )
            {
                for ( int x = 0 ; x < w ; x++ )
                {
                    if ( edge[i].at<uchar>(y, x) > 0 && iVar[i].at<float>(y, x) > 0 ){
                        cnt++ ;
                    }
                }
            }
            edge_2d[i] = Eigen::MatrixXf::Zero(2, cnt);
            edge_3d[i] = Eigen::MatrixXf::Zero(3, cnt);

            cnt = 0 ;
            float X, Y, Z;
            for( int y = 0 ; y < h ; y++ )
            {
                for ( int x = 0 ; x < w ; x++ )
                {
                    if ( edge[i].at<uchar>(y, x) > 0 && iVar[i].at<float>(y, x) > 0 )
                    {
                        edge_2d[i](0, cnt) = x ;
                        edge_2d[i](1, cnt) = y ;

                        Z = 1.0 / idepth[i].at<float>(y, x) ;
                        X = ( x - cali->cx[i] ) / cali->fx[i] * Z ;
                        Y = ( y - cali->cy[i] ) / cali->fy[i] * Z ;

                        edge_3d[i](0, cnt) = X ;
                        edge_3d[i](1, cnt) = Y ;
                        edge_3d[i](2, cnt) = Z ;
                        cnt++ ;
                    }
                }
            }
        }
    }
    /*
    void initPointsPyramid(CALIBRATION_PAR* cali)
    {
        if ( makePointAndBuffer ){
            return ;
        }
        makePointAndBuffer = true ;
        int w, h ;
        for( int i = 0 ; i < optimizedPyramidLevel ; i++ )
        {
            w = edge[i].cols ;
            h = edge[i].rows ;

            if ( posData[i] != NULL ){
                delete[] posData[i];
                posData[i] = NULL ;
            }
            if ( posVar[i] != NULL ){
                Eigen::internal::aligned_free((void*)posVar[i]);
                posVar[i] = NULL ;
            }
            posData[i] = new Eigen::Vector3f[w*h];
            posVar[i] = (float*)Eigen::internal::aligned_malloc(w*h*sizeof(float));

            //count the number of selected pixels
            int cnt = 0 ;
            Eigen::Vector3f* pData = posData[i] ;
            float* pVar = posVar[i] ;
            float X, Y, Z;
            for( int y = 0 ; y < h ; y++ )
            {
                for ( int x = 0 ; x < w ; x++ )
                {
                    if ( edge[i].at<uchar>(y, x) > 0 && iVar[i].at<float>(y, x) > 0 )
                    {
                        Z = 1.0 / idepth[i].at<float>(y, x) ;
                        X = ( x - cali->cx[i] ) / cali->fx[i] * Z ;
                        Y = ( y - cali->cy[i] ) / cali->fy[i] * Z ;
                        *pData << X, Y, Z ;
                        *pVar = iVar[i].at<float>(y, x) ;
                        pData++ ;
                        pVar++ ;
                        cnt++ ;
                    }
                }
            }
            numOfposData[i] = cnt ;
        }
    }
    */
    void initEdgePyramid()
    {
        cv::Mat output ;
        double threshold ;
        for( int i = 0 ; i < optimizedPyramidLevel ; i++ )
        {
            if ( enable_histogramEqualization ){
                cv::equalizeHist(image[i], image[i]) ;
            }
            if ( adaptiveCannyThreshold ){
                threshold = cv::threshold(image[i], output, 0, 255, cv::THRESH_OTSU ) ;
                cv::Canny(image[i], edge[i], std::max(0, (int)threshold-10 ), std::min(255, (int)threshold+10), 3, true ) ;
            }
            else{
                cv::Canny(image[i], edge[i], cannyThreshold1, cannyThreshold2, 3, true ) ;
            }
            //            cv::Canny(image[i], edge[i], max(0, cannyThreshold1, cannyThreshold2, 3, true ) ;
            //            printf("cannyThreshold1 = %d", cannyThreshold1 ) ;
            //            printf("cannyThreshold2 = %d", cannyThreshold2 ) ;
            //edge[i] = 255 - edge[i] ;
            edgeNum[i] = cv::countNonZero(edge[i]) ;
        }
    }

    void initDistanceTransformPyramid()
    {
        cv::Mat kernX = (cv::Mat_<float>(3,3) <<  0, 0,  0,
                         -0.5,  0.0, .5,
                         0, 0,  0);
        cv::Mat kernY = (cv::Mat_<float>(3,3) <<  0, -0.5,  0,
                         0,  0.0, 0,
                         0, 0.5,  0);

        cv::Mat gradientX ;
        cv::Mat gradientY ;
        for( int i = 0 ; i < optimizedPyramidLevel ; i++ )
        {
            cv::distanceTransform(255-edge[i], distanceTransformMap[i], CV_DIST_L2, CV_DIST_MASK_PRECISE) ;
            cv::cv2eigen(distanceTransformMap[i], distanceTransformEigen[i]) ;

            cv::filter2D( distanceTransformMap[i], gradientX, CV_32F, kernX );
            cv::cv2eigen( gradientX, distanceTransformGradientX[i] ) ;
            cv::filter2D( distanceTransformMap[i], gradientY, CV_32F, kernY );
            cv::cv2eigen( gradientY, distanceTransformGradientY[i] ) ;
        }
    }

    STATE()
    {

        //set the keyFrameFlag
        keyFrameFlag = false;
        cameraLink.clear();

        makePointAndBuffer = false ;
        for( int i = 0 ; i < PYRAMID_LEVELS ; i++ ){
            posData[i] = NULL ;
            posVar[i] = NULL ;
        }
        //is this state ready?
        isReady = false ;

        iDepthMean = 0 ;
        frame = nullptr;
    }

    STATE(ros::Time time, cv::Mat img, cv::Mat idep, cv::Mat iV)
    {
        tImage = time ;

        //set the keyFrameFlag
        keyFrameFlag = false;
        isReady = false ;

        //reset the camera link list
        cameraLink.clear();

        image[0] = img.clone() ;
        idepth[0] = idep.clone() ;
        iVar[0] = iV.clone() ;

        initPyramid() ;

        makePointAndBuffer = false ;
        for( int i = 0 ; i < PYRAMID_LEVELS ; i++ ){
            posData[i] = NULL ;
            posVar[i] = NULL ;
        }
        frame = nullptr;
    }

    void reset(ros::Time time, cv::Mat img )
    {
        tImage = time ;

        //set the keyFrameFlag
        keyFrameFlag = false;

        //reset the camera link list
        cameraLink.clear();

        image[0] = img.clone() ;
        //idepth[0] = idep.clone() ;
        //iVar[0] = iV.clone() ;

        initPyramid() ;

        isReady = true ;

        makePointAndBuffer = false ;
        for( int i = 0 ; i < PYRAMID_LEVELS ; i++ )
        {
            if ( posData[i] != NULL ){
                delete[] posData[i];
                posData[i] = NULL ;
            }
            if ( posVar[i] != NULL ){
                Eigen::internal::aligned_free((void*)posVar[i]);
                posVar[i] = NULL ;
            }
        }
        frame = nullptr;
    }

    void insertDepth( cv::Mat idep, cv::Mat iV)
    {
        idepth[0] = idep.clone() ;
        iVar[0] = iV.clone() ;
        initDepthPyramid();

        int width = idepth[0].cols ;
        int height = idepth[0].rows ;
        //depth and variance
        double idepthSumsSum = 0;
        double ivarSumsSum = 0;
        double var ;
        double ivar;
        int num=0;
        for(int y=0; y<height; y++)
        {
            for(int x=0; x<width; x++ )
            {
                var = iVar[0].at<float>( y, x ) ;
                if(var > 0)
                {
                    ivar = 1.0f / var;
                    ivarSumsSum += ivar;
                    idepthSumsSum += ivar * idepth[0].at<float>( y, x );
                    num++;
                }
            }
        }

        if(num > 0){
            iDepthMean = idepthSumsSum / ivarSumsSum ;
        }
        else{
            iDepthMean = -1.0 ;
        }
    }

    ~STATE()
    {
        for( int i = 0 ; i < PYRAMID_LEVELS ; i++ )
        {
            if ( posData[i] != NULL ){
                delete[] posData[i];
                posData[i] = NULL ;
            }
            if ( posVar[i] != NULL ){
                Eigen::internal::aligned_free((void*)posVar[i]);
                posVar[i] = NULL ;
            }
        }    ;
    }
};

struct CAMERALINK
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix3d R_bi_2_bj;//rotation from current body to link body
    Eigen::Vector3d T_bi_2_bj;//translation from current body to link body
    Eigen::Matrix<double, 6, 6> P_inv;
    STATE* pState ;
    CAMERALINK(){
        P_inv.setIdentity() ;
        pState = NULL ;
    }
};
