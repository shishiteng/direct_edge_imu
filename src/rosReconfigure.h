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

#include <dynamic_reconfigure/server.h>
#include "edge_imu_bias/ParamsConfig.h"
#include "settings.h"


void dynConfCb(edge_imu_bias::ParamsConfig &config, uint32_t level)
{
    gaussianNewtonTrackingIterationNum = config.gaussianNewtonIterationNum ;
    optimizedPyramidLevel = config.pyramidLevelInUse ;
    printDebugInfo = config.printDebugInfo ;
    useGaussianNewton = config.enableGaussianNewtonOptimization ;
    visualizeTrackingDebug = config.visualizeTrackingDebug ;
    visaulizeGraphStructure = config.visaulizeGraphStructure ;
    enable_LoopClosure = config.enableLoopClosure ;
    enable_pubPointCloud = config.pubPointCloud ;
    enable_pubKeyFrameOdom = config.pubKeyFrameOdom ;
    enable_pubTF = config.pubTF ;
    subGradientTrackingIterationNum = config.subGradientIterationNum ;
    enable_crossCheckTracking = config.enableCrossCheckTracking ;
    trustTrackingErrorThreshold = config.trustTrackingErrorThreshold  ;
    cannyThreshold1 = config.cannyThreshold1 ;
    cannyThreshold2 = config.cannyThreshold2 ;

    KFDistWeight = config.KFDistWeight;
    KFUsageWeight = config.KFUsageWeight;
    minUseGrad = config.minUseGrad;
    cameraPixelNoise2 = config.cameraPixelNoise*config.cameraPixelNoise;
    maxLoopClosureCandidates = config.maxLoopClosureCandidates;

    enable_histogramEqualization = config.enable_histogramEqualization ;
    edgeProportion = config.edgeProportion ;
}
