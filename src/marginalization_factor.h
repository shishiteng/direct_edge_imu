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

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>

#include <ceres/ceres.h>
#include <unordered_map>

#include "utility.h"
#include "tic_toc.h"

struct ResidualBlockInfo
{
    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;

    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    int localSize(int size){
        return size == 7 ? 6 : size;
    }

    void Evaluate()
    {
        residuals.resize(cost_function->num_residuals());

        std::vector<int> block_sizes = cost_function->parameter_block_sizes();
        raw_jacobians = new double *[block_sizes.size()];
        jacobians.resize(block_sizes.size());

        //int dim = 0;
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
            raw_jacobians[i] = jacobians[i].data();
            //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
        }
        cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

        //std::vector<int> tmp_idx(block_sizes.size());
        //Eigen::MatrixXd tmp(dim, dim);
        //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        //{
        //    int size_i = localSize(block_sizes[i]);
        //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
        //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
        //    {
        //        int size_j = localSize(block_sizes[j]);
        //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
        //        tmp_idx[j] = sub_idx;
        //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
        //    }
        //}
        //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
        //std::cout << saes.eigenvalues() << std::endl;
        //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);

        if (loss_function)
        {
            double residual_scaling_, alpha_sq_norm_;

            double sq_norm, rho[3];

            sq_norm = residuals.squaredNorm();
            loss_function->Evaluate(sq_norm, rho);
            //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

            double sqrt_rho1_ = sqrt(rho[1]);

            if ((sq_norm == 0.0) || (rho[2] <= 0.0))
            {
                residual_scaling_ = sqrt_rho1_;
                alpha_sq_norm_ = 0.0;
            }
            else
            {
                const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
                const double alpha = 1.0 - sqrt(D);
                residual_scaling_ = sqrt_rho1_ / (1 - alpha);
                alpha_sq_norm_ = alpha / sq_norm;
            }

            for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
            {
                jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
            }

            residuals *= residual_scaling_;
        }
    }
};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    int localSize(int size) const;
    int globalSize(int size) const;
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    std::vector<std::shared_ptr<ResidualBlockInfo>> factors;
    int m, n;
    std::unordered_map<long, int> parameter_block_size; //global size
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size
    std::unordered_map<long, double *> parameter_block_data;

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
};
