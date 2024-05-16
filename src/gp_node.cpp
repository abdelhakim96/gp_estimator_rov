#include "gp_node.h" // Include GPKit library


class GP : public rclcpp::Node {
public:
   double rate_gp_ = 10;
   int input_size_ = 10;

    GP() : Node("gp_node") {
   

        rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 10), qos_profile);
 

        // Create a subscribers 
        odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/mobula/rov/odometry", qos, std::bind(&GP::odom_cb, this, std::placeholders::_1));

        accel_subscription_ = this->create_subscription<geometry_msgs::msg::Vector3Stamped>(
            "/mobula/rov/acceleration", qos, std::bind(&GP::accel_cb, this, std::placeholders::_1));

        mpc_subscription_ = this->create_subscription<geometry_msgs::msg::Wrench>(
            "/mobula/rov/wrench", qos, std::bind(&GP::mpc_cb, this, std::placeholders::_1));


       //Create publisher
         gp_pred_mu_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>
                      ("/gp_pred_mu", 10);   //change

         gp_pred_cov_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>
              ("/gp_pred_cov", 10);
              



        // Initialize the Gaussian Process with hyperparameters
        Eigen::VectorXd theta_0 = Eigen::VectorXd::Ones(5);

        gp_x_ = GaussianProcess(theta_0);
        gp_y_ = GaussianProcess(theta_0);
        gp_z_ = GaussianProcess(theta_0);
     

        publish_mu_pred();

    }



    std::atomic<int> prediction_counter_;



    void predictGPX() {
        while (rclcpp::ok()) {
            // Extract states for GP_x prediction (assuming current_states is available)
            // Use first 3 states for GP_x input
            // ...
             Eigen::MatrixXd gp_x_Xdata(1, 4);
             Eigen::MatrixXd gp_x_Xpred(1, 4);
             Eigen::MatrixXd gp_x_Ydata(1, 1);
             
            

	         gp_x_Xdata(0, 0) = current_states.at(0);  
	         gp_x_Xdata(0, 1) = current_states.at(3);
	         gp_x_Xdata(0, 2) = current_states.at(7);
	         gp_x_Xdata(0, 3) = mpc_command.at(0);

	         gp_x_Xpred(0, 0) = current_states.at(0);
	         gp_x_Xpred(0, 1) = current_states.at(3);
	         gp_x_Xpred(0, 2) = current_states.at(7);
	         gp_x_Xpred(0, 3) = mpc_command.at(0);

            std::cout << "gp" <<  gp_x_Xdata(0, 0);


	         gp_x_Ydata(0, 0) = acceleration.at(0) - current_states.at(4) * mpc_command.at(3) +
	            mpc_command.at(2) * current_states.at(5) - 9.81 * sin(current_states.at(7));


             gp_x_.add_sample(gp_x_Ydata(0, 0), gp_x_Xdata);
             
             

             std::tie(mu_x, cov_x) = gp_x_.predict(gp_x_Xpred);


            if (gp_x_.getDataSize()>input_size_){

            	gp_x_.remove_sample();
            }


            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Adjust as needed



                    this->prediction_counter_++;
        }
    }


     void predictGPY() {
        while (rclcpp::ok()) {
            // Extract states for GP_y prediction (assuming current_states is available)
            // ...
             Eigen::MatrixXd gp_y_Xdata(1, 4);
             Eigen::MatrixXd gp_y_Xpred(1, 4);
             Eigen::MatrixXd gp_y_Ydata(1, 1);


	         gp_y_Xdata(0, 0) = current_states.at(1);
	         gp_y_Xdata(0, 1) = current_states.at(4);
	         gp_y_Xdata(0, 2) = current_states.at(6);
	         gp_y_Xdata(0, 3) = mpc_command.at(0);

	         gp_y_Xpred(0, 0) = current_states.at(1);
	         gp_y_Xpred(0, 1) = current_states.at(4);
	         gp_y_Xpred(0, 2) = current_states.at(6);
	         gp_y_Xpred(0, 3) = mpc_command.at(0);



	         gp_y_Ydata(0, 0) = acceleration.at(1) - current_states.at(5) * mpc_command.at(1) +
	            mpc_command.at(3) * current_states.at(3) + 9.81 * sin(current_states.at(6)) * cos(current_states.at(7));

             gp_y_.add_sample(gp_y_Ydata(0, 0), gp_y_Xdata);
            

             std::tie(mu_y, cov_y) = gp_y_.predict(gp_y_Xpred);

         	 
             
             if (gp_y_.getDataSize()>input_size_){

            	gp_y_.remove_sample();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Adjust as needed

                    this->prediction_counter_++;
        }




    }

         void predictGPZ() {
        while (rclcpp::ok()) {
            // Extract states for GP_y prediction (assuming current_states is available)
            // ...
             Eigen::MatrixXd gp_z_Xdata(1, 4);
             Eigen::MatrixXd gp_z_Xpred(1, 4);
             Eigen::MatrixXd gp_z_Ydata(1, 1);


	         gp_z_Xdata(0, 0) = current_states.at(2);
	         gp_z_Xdata(0, 1) = current_states.at(6);
	         gp_z_Xdata(0, 2) = current_states.at(7);
	         gp_z_Xdata(0, 3) = mpc_command.at(0);

	         gp_z_Xpred(0, 0) = current_states.at(6);
	         gp_z_Xpred(0, 1) = current_states.at(6);
	         gp_z_Xpred(0, 2) = current_states.at(7);
	         gp_z_Xpred(0, 3) = mpc_command.at(0);




	         gp_z_Ydata(0, 0) = acceleration.at(2) - current_states.at(3) * mpc_command.at(2) +
	            mpc_command.at(1) * current_states.at(4) + 
	            9.81 * cos( current_states.at(6)) * cos(current_states.at(7)) -
	             (1 / 1.2) * (mpc_command.at(0));

             gp_z_.add_sample(gp_z_Ydata(0, 0), gp_z_Xdata);
            
             
             std::tie(mu_z, cov_z) = gp_z_.predict(gp_z_Xpred);

            
             if (gp_z_.getDataSize() > input_size_){

            	gp_z_.remove_sample();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Adjust as needed

                    this->prediction_counter_++;
        }
    }


    void publish_mu_pred() {
        std_msgs::msg::Float64MultiArray msg;
        msg.data = {mu_x(0, 0), mu_y(0, 0), mu_z(0, 0)};
        gp_pred_mu_publisher_->publish(msg);
    }

    void publish_cov_pred(const double& cov_x, const double& cov_y, const double& cov_z) {
        std_msgs::msg::Float64MultiArray msg;
        msg.data = {cov_x, cov_y, cov_z};
        gp_pred_cov_publisher_->publish(msg);
    }

private:


  
    void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg);
    void accel_cb(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg);
    void mpc_cb(const geometry_msgs::msg::Wrench::SharedPtr msg);


    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr accel_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::Wrench>::SharedPtr mpc_subscription_;


    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr gp_pred_mu_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr gp_pred_cov_publisher_;


    GaussianProcess gp_x_;
    GaussianProcess gp_y_;
    GaussianProcess gp_z_;
    

};






void GP::mpc_cb(const geometry_msgs::msg::Wrench::SharedPtr msg) {

    mpc_command = {  msg->force.x,
    	              msg->force.y, 
                      msg->force.z,
                      msg->torque.z};
}



void GP::accel_cb(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {
    acceleration = {msg->vector.x, msg->vector.y, msg->vector.z};



}

void GP::odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg) {  
        // Extract data from Odometry message
    tf2::Quaternion q(
       msg->pose.pose.orientation.x,
       msg->pose.pose.orientation.y,
       msg->pose.pose.orientation.z,
       msg->pose.pose.orientation.w);

    tf2::Matrix3x3 m(q);

    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    double vx     = msg->twist.twist.linear.x;
    double vy     = msg->twist.twist.linear.y;
    double vz     = msg->twist.twist.linear.z;
    double Phi    = yaw;
    double Theta  = pitch;
    double Psi    = roll-1.570;


    Psi = -Psi;
  

    Eigen::Matrix3d Rx, Ry, Rz;

    Rx << 1, 0, 0,
          0, cos(Psi), -sin(Psi),
          0, sin(Psi), cos(Psi);

    Ry << cos(Psi), 0, sin(Psi),
          0, 1, 0,
          -sin(Psi), 0, cos(Psi);

    Rz << cos(Psi), -sin(Psi), 0,
          sin(Psi), cos(Psi), 0,
          0, 0, 1;


    Eigen::Matrix3d R = Rz * Ry * Rx;

    R = R.transpose().eval();
   
    double r_vx = R(0, 0) * vx + R(0, 1) * vy + R(0, 2) * vz;
    double r_vy = R(1, 0) * vx + R(1, 1) * vy + R(1, 2) * vz;
    double r_vz = R(2, 0) * vx + R(2, 1) * vy + R(2, 2) * vz;


  current_states = { msg->pose.pose.position.x, 
                      msg->pose.pose.position.y,
                      msg->pose.pose.position.z,
                          r_vx,
                          r_vy,
                          r_vz,
                         yaw,    //roll
                          pitch,  //pitch
                          -Psi};   //yaw

    }








int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
     auto gp_node = std::make_shared<GP>();


     std::thread threadGPX(&GP::predictGPX, gp_node);
     std::thread threadGPY(&GP::predictGPY, gp_node);
     std::thread threadGPZ(&GP::predictGPZ, gp_node);


        while (rclcpp::ok()){ 

       //     std::cout << "Hakim";
        //    std::cout << "mpc" << mpc_command.at(0);
        //    std::cout << "x" << mpc_command.at(0);

        while (gp_node->prediction_counter_ < 3) {
		    std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Adjust as needed

		}
        
        gp_node->publish_mu_pred();
        gp_node->prediction_counter_ = 0; // Reset the counter
        rclcpp::spin_some(gp_node);

	    
	}

	    threadGPX.join();
        threadGPY.join();
        threadGPZ.join();

	    rclcpp::shutdown();
	    return 0;
}