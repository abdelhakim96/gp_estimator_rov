
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
#include "std_msgs/msg/bool.hpp"
#include <iostream>
#include <future>
#include <thread>
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/wrench.hpp"
#include "gp_fast_sgp.h"
#include "gp_sgp.h"

// Global variables
rclcpp::Time start_time;
std::vector<double> current_states(9, 0.0);  // Initialized vector of size 9 with zeros
std::vector<double> ref_traj(9, 0.0);        // Initialized vector of size 9 with zeros
std::vector<double> control(4, 0.0);     // Initialized vector of size 4 with zeros
std::vector<double> acceleration(3, 0.0);    // Initialized vector of size 3 with zeros
std::vector<double> velocity(3, 0.0);    // Initialized vector of size 3 with zeros
std::vector<double> position(3, 0.0);    // Initialized vector of size 3 with zeros

double roll, pitch, yaw;
double Fx_dist , Fy_dist;
double Fx_dist_t=0;
double Fy_dist_t=0;

// BlueROV2 Model Parameters
const double f_s = 1.0;    //scaling factor;
const double m = 13.4;    // BlueROV2 mass (kg)  
const double g = 9.82;  // gravitational field strength (m/s^2)
const double F_bouy = 114.8; // Buoyancy force (N)
 
const double X_ud = -2.6 ; // Added mass in x direction (kg)
const double Y_vd = -18.5 ; // Added mass in y direction (kg)
const double Z_wd = -13.3 ; // Added mass in z direction (kg)
const double N_rd = -0.28 ; // Added mass for rotation about z direction (kg)
 
const double I_xx = 0.21 ; // Moment of inertia (kg.m^2)
const double I_yy = 0.245 ; // Moment of inertia (kg.m^2)
const double I_zz = 0.245 ; // Moment of inertia (kg.m^2)
 
const double X_u = -0.09 ; // Linear damping coefficient in x direction (N.s/m)
const double Y_v  = -0.26 ; // Linear damping coefficient  in y direction (N.s/m)
const double Z_w = -0.19; // Linear damping coefficient  in z direction (N.s/m)
const double N_r = -4.64 ;  // Linear damping coefficient for rotation about z direction (N.s/rad)
 
const double X_uc = -34.96 ; // quadratic damping coefficient in x direction (N.s^2/m^2)
const double Y_vc = -103.25 ; // quadratic damping coefficient  in y direction (N.s^2/m^2)
const double Z_wc = -74.23 ; // quadratic damping coefficient  in z direction (N.s^2/m^2)
const double N_rc = - 0.43 ; // quadratic damping coefficient for rotation about z direction (N.s^2/rad^2)


// Gaussian Process variables
Eigen::MatrixXd mu_x = Eigen::MatrixXd::Zero(1, 1);
double cov_x = 0.0;

Eigen::MatrixXd mu_y = Eigen::MatrixXd::Zero(1, 1);
double cov_y = 0.0;

Eigen::MatrixXd mu_z = Eigen::MatrixXd::Zero(1, 1);
double cov_z = 0.0;
bool traj_on = 0;
bool opt_x_done = 0;
bool opt_y_done = 0;
bool opt_z_done = 0;
