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
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "ros/ros.h"
#include "tf/transform_broadcaster.h"
#include "nav_msgs/Odometry.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"
#include "nav_msgs/Path.h"
#include "visualization_msgs/Marker.h"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.h"
#include "sensor_msgs/image_encodings.h"
#include "sensor_msgs/PointCloud.h"
#include "sensor_msgs/fill_image.h"
#include "quadrotor_msgs/Odometry.h"
#include "settings.h"

using namespace Eigen;
static int keyframeIDCount = 0 ;
static nav_msgs::Odometry last_kfodom ;

inline void pubOdometry(const Vector3d& p, const Vector3d& vel, const Matrix3d& R,
                        ros::Publisher& pub_odometry, ros::Publisher& pub_pose,
                        ros::Publisher& pub_navOdometry,
                        int control_flag, const Matrix3d& R_vi_2_odometry, bool keyframeFlag, ros::Time tImage )
{
  quadrotor_msgs::Odometry output_odometry ;
  keyframeIDCount += keyframeFlag ;

  nav_msgs::Odometry odometry;
  Vector3d output_p = R_vi_2_odometry * p;
  Vector3d output_v = R_vi_2_odometry * vel;
  Matrix3d output_R = R_vi_2_odometry * R * R_vi_2_odometry.transpose();

  Eigen::Quaterniond q(output_R) ;


  odometry.header.stamp = tImage;
  odometry.header.frame_id = "world";
  odometry.pose.pose.position.x = output_p(0);
  odometry.pose.pose.position.y = output_p(1);
  odometry.pose.pose.position.z = output_p(2);
  odometry.pose.pose.orientation.x = q.x();
  odometry.pose.pose.orientation.y = q.y();
  odometry.pose.pose.orientation.z = q.z();
  odometry.pose.pose.orientation.w = q.w();
  odometry.twist.twist.linear.x = output_v(0) ;
  odometry.twist.twist.linear.y = output_v(1) ;
  odometry.twist.twist.linear.z = output_v(2) ;

  if ( control_flag == 0 ){
    //odometry.child_frame_id = "V" ;
    output_odometry.status = quadrotor_msgs::Odometry::STATUS_ODOM_VALID ;
  }
  else if ( control_flag == 1 ){
    //odometry.child_frame_id = "L" ;
    output_odometry.status = quadrotor_msgs::Odometry::STATUS_ODOM_LOOPCLOSURE ;
  }
  else {
    //odometry.child_frame_id = "X" ;
    output_odometry.status = quadrotor_msgs::Odometry::STATUS_ODOM_INVALID ;
  }
  output_odometry.curodom = odometry ;


#ifdef   PUB_KEYFRAME_ODOM
  output_odometry.kfid = keyframeIDCount ;
  if ( keyframeFlag ){
    output_odometry.kfodom = odometry ;
    last_kfodom = odometry ;
  }
  else {
    output_odometry.kfodom = last_kfodom ;
  }
#else
  nav_msgs::Odometry kfodom;
  output_odometry.kfid = 1 ;
  kfodom.header.stamp = tImage;
  kfodom.header.frame_id = "world";
  kfodom.pose.pose.position.x = 0;
  kfodom.pose.pose.position.y = 0;
  kfodom.pose.pose.position.z = 0;
  kfodom.pose.pose.orientation.x = 0;
  kfodom.pose.pose.orientation.y = 0;
  kfodom.pose.pose.orientation.z = 0;
  kfodom.pose.pose.orientation.w = 1.0;
  kfodom.twist.twist.linear.x = 0 ;
  kfodom.twist.twist.linear.y = 0 ;
  kfodom.twist.twist.linear.z = 0 ;
  output_odometry.kfodom = kfodom ;
#endif

  pub_navOdometry.publish(odometry) ;

  pub_odometry.publish(output_odometry);

  geometry_msgs::PoseStamped pose_stamped;
  pose_stamped.header.stamp = tImage;
  pose_stamped.header.frame_id = "world";
  pose_stamped.pose = odometry.pose.pose;
  pub_pose.publish(pose_stamped);


}

inline void pubPath(const Vector3d& p,
                    int kind,
                    visualization_msgs::Marker& path_line,
                    ros::Publisher& pub_path, const Matrix3d& R_vi_2_odometry )
{
    geometry_msgs::Point pose_p;
    std_msgs::ColorRGBA color_p ;
    Vector3d ttp = R_vi_2_odometry*p ;
    pose_p.x = ttp(0);
    pose_p.y = ttp(1);
    pose_p.z = ttp(2);
    if ( kind == 0 ){
        color_p.r = 1.0 ;
        color_p.g = 0.0 ;
        color_p.b = 0.0 ;
        color_p.a = 1.0 ;
    }
    else if ( kind == 1 )
    {
        color_p.r = 0.0 ;
        color_p.g = 1.0 ;
        color_p.b = 0.0 ;
        color_p.a = 1.0 ;
    }
    else{
      color_p.r = 1.0 ;
      color_p.g = 1.0 ;
      color_p.b = 0.0 ;
      color_p.a = 1.0 ;
    }
    path_line.colors.push_back(color_p);
    path_line.points.push_back(pose_p);
    path_line.scale.x = 0.01 ;
    pub_path.publish(path_line);
}
