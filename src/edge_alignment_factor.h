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
#include <ros/assert.h>
#include <iostream>
#include <Eigen/Dense>

#include "utility.h"
#include "myMath.h"
#include "parameters.h"
#include "settings.h"
#include "globalFuncs.h"
#include <ceres/ceres.h>

class EdgeAlignmentFactor : public ceres::SizedCostFunction<1, 7>
{
public:
    EdgeAlignmentFactor() = delete ;
    EdgeAlignmentFactor( double x, double y, double z,
                         Eigen::MatrixXf& map, Eigen::MatrixXf& dx, Eigen::MatrixXf&dy,
                         double fx, double fy, double cx, double cy, int w, int h)
        :fx(fx), fy(fy), cx(cx), cy(cy), dGx(dx), dGy(dy), cost(map)
    {
        width = w - 3 ;
        height = h - 3 ;
        p << x, y, z ;
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d t(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Matrix3d R = q.toRotationMatrix();
        //Eigen::Matrix3d Ri = Qi.toRotationMatrix() ;

//        std::cout << "t = " << t.transpose() << "\n" ;
//        std::cout << "R = \n" << R << "\n" ;


        Eigen::Vector3d p2 = R*p + t ;
        float X = (float)p2(0);
        float Y = (float)p2(1);
        float Z = (float)p2(2);
        bool out = false ;

        float u_ = fx * X / Z + cx ;
        float v_ = fy * Y / Z + cy ;
        //printf("u_%f v_%f\n", u_, v_) ;
        if (u_ < 2 || u_ > width || v_ < 2 || v_ > height || Z < 0.0001 ){
            residuals[0] = 0 ;
            out = true;
        }
        else {
            residuals[0] = (double)getInterpolatedElementEigen(cost, u_, v_) ;
        }
        //ROS_WARN("%lf\n", residuals[0] ) ;
        if ( jacobians )
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero() ;
                if ( out == false )
                {
                    float gx = getInterpolatedElementEigen(dGx, u_, v_) ;
                    float gy = getInterpolatedElementEigen(dGy, u_, v_) ;
                    //ROS_WARN("gx=%f gy=%f\n", gx, gy ) ;

                    Eigen::Matrix<double,1,3> oneByThree;
                    oneByThree(0, 0) = gx*fx/Z;
                    oneByThree(0, 1) = gy*fy/Z;
                    oneByThree(0, 2) = -gx*fx*X/(Z*Z) - gy*fy*Y/(Z*Z);

                    Eigen::Matrix3d threeByThree = -R*  Utility::skewSymmetric(p) ;

                    jacobian_pose_i(0, 0) = oneByThree(0, 0) ;
                    jacobian_pose_i(0, 1) = oneByThree(0, 1) ;
                    jacobian_pose_i(0, 2) = oneByThree(0, 2) ;
                    jacobian_pose_i.block<1, 3>(0, 3) = oneByThree*threeByThree ;

                    //std::cout << "oneByThree :" << oneByThree << "\n" ;
                    //std::cout << "threeByThree :" << threeByThree << "\n" ;
                    //std::cout << "jacobian_pose_i " << jacobian_pose_i << "\n";

                    if (fabs(jacobian_pose_i.maxCoeff()) > 1e8 ||
                            fabs(jacobian_pose_i.minCoeff()) < -1e8)
                    {
                        std::cout << "oneByThree :" << oneByThree << "\n" ;
                        std::cout << "threeByThree :" << threeByThree << "\n" ;
                        std::cout << "jacobian_pose_i " << jacobian_pose_i << "\n";
                        ROS_BREAK();
                    }
                }
            }
        }

        return true;
    }

    Eigen::Vector3d p ;
    double fx, fy, cx, cy;
    int width;
    int height;
    Eigen::MatrixXf& cost;
    Eigen::MatrixXf& dGx;
    Eigen::MatrixXf& dGy;
};
