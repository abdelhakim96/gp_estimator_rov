
/*
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_rates_setpoint.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/trajectory_setpoint.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include "px4_msgs/msg/sensor_accel.hpp"
#include <px4_msgs/msg/vehicle_control_mode.hpp>
*/

#include <rclcpp/rclcpp.hpp>
#include <stdint.h>
#include "tf2/transform_datatypes.h"
#include "tf2/utils.h"
#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include "gaussian_process.h" // Include GPKit library
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include <iostream>
#include <future>
#include <thread>
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/wrench.hpp"

// Global variables
rclcpp::Time start_time;
std::vector<double> current_states(9, 0.0);  // Initialized vector of size 9 with zeros
std::vector<double> ref_traj(9, 0.0);        // Initialized vector of size 9 with zeros
std::vector<double> mpc_command(4, 0.0);     // Initialized vector of size 4 with zeros
std::vector<double> acceleration(3, 0.0);    // Initialized vector of size 3 with zeros

double roll, pitch, yaw;

// Gaussian Process variables
Eigen::MatrixXd mu_x = Eigen::MatrixXd::Zero(1, 1);
double cov_x = 0.0;

Eigen::MatrixXd mu_y = Eigen::MatrixXd::Zero(1, 1);
double cov_y = 0.0;

Eigen::MatrixXd mu_z = Eigen::MatrixXd::Zero(1, 1);
double cov_z = 0.0;


