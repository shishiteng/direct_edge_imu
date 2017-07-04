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
#include <ceres/ceres.h>

class RelativePoseFactor : public ceres::SizedCostFunction<6, 7, 7>
{
  public:
    RelativePoseFactor(const Eigen::Matrix3d &R, const Eigen::Vector3d &T, const Eigen::Matrix<double, 6, 6>& P_inv )
    {
        R_k_2_c = R ;
        T_k_2_c = T ;
        Cov_inv = P_inv ;
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        //Eigen::Matrix3d Ri = Qi.toRotationMatrix() ;

        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();

        Eigen::Quaterniond q_i_2_j(R_k_2_c) ;

        Eigen::Matrix<double, 6, 6> tmpP_inv;
        tmpP_inv = Cov_inv;

        Vector3d r_p = Rj.transpose()*(Pi - Pj) - T_k_2_c;
        //Vector3d r_q = Quaterniond(R_k_2_c.transpose() * Rj.transpose() * Ri).vec() * 2.0 ;
        Vector3d r_q = (q_i_2_j.inverse()*Qj.inverse()*Qi).vec()*2.0 ;

        double r_v = r_p.norm() ;
        if ( r_v > huber_r_v ){
            tmpP_inv *= huber_r_v/r_v ;
        }
        double r_w = r_q.norm() ;
        if ( r_w > huber_r_w ){
            tmpP_inv *= huber_r_w/r_w ;
        }
        Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(tmpP_inv).matrixL().transpose();

        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
        residual.block<3, 1>(O_P, 0) = r_p;
        residual.block<3, 1>(O_R, 0) = r_q;
        residual = sqrt_info * residual;
        //std::cout << "[RP] " << residual.transpose() << "\n" ;

        if (jacobians)
        {
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(O_P, O_P) = Rj.transpose();
                //jacobian_pose_i.block<3, 3>(O_R, O_R).setIdentity();
                jacobian_pose_i.block<3, 3>(O_R, O_R) = Utility::Qleft(q_i_2_j.inverse() * Qj.inverse() * Qi).bottomRightCorner<3, 3>();

                jacobian_pose_i = sqrt_info * jacobian_pose_i;

                //std::cout << "[RP] " << jacobian_pose_i << "\n" ;

                if (fabs(jacobian_pose_i.maxCoeff()) > 1e8 ||
                        fabs(jacobian_pose_i.minCoeff()) < -1e8)
                {
                    std::cout << jacobian_pose_i << std::endl;
                    ROS_BREAK();
                }
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(O_P, O_P) = -Rj.transpose();
                //jacobian_pose_j.block<3, 3>(O_P, O_R) = math.Skew( Rj.transpose()*(Pi-Pj) );
                jacobian_pose_j.block<3, 3>(O_P, O_R) = Utility::skewSymmetric(Qj.inverse()*(Pi-Pj) );
                //jacobian_pose_j.block<3, 3>(O_R, O_R) = -Ri.transpose()*Rj;
                jacobian_pose_j.block<3, 3>(O_R, O_R) =
                        -(Utility::Qleft(Qi.inverse() * Qj) * Utility::Qright(q_i_2_j)).bottomRightCorner<3, 3>();

                jacobian_pose_j = sqrt_info * jacobian_pose_j;

                //std::cout << "[RP] " << jacobian_pose_j << "\n" ;

                if (fabs(jacobian_pose_j.maxCoeff()) > 1e8 ||
                        fabs(jacobian_pose_j.minCoeff()) < -1e8)
                {
                    std::cout << jacobian_pose_j << std::endl;
                    ROS_BREAK();
                }
            }
        }

        return true;
    }

    Eigen::Matrix3d R_k_2_c ;
    Eigen::Vector3d T_k_2_c ;
    Eigen::Matrix<double, 6, 6> Cov_inv ;
};
