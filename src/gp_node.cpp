#include "gp_node.h" // Include GPKit library


class GP : public rclcpp::Node {
public:
   double rate_gp_ = 10;
   int input_size_ = 2000;
   double lambda_ = 0.97;

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


         gp_pred_mu_x_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>
                      ("/gp_disturb_reg/mu/x", 10);   

         gp_pred_mu_y_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>
                      ("/gp_disturb_reg/mu/y", 10);   

        gp_pred_mu_z_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>
                      ("/gp_disturb_reg/mu/z", 10);             


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
             
            
             //calculate disturbance estimate in x

             Fx_dist = (m - X_ud) * acceleration[0] - (control[0] +
                (m * velocity[1] -Y_vd * velocity[1]) * -velocity[5] +
                (X_u + X_uc * sqrt(velocity[0] * velocity[0])) * velocity[0]);

             Fx_dist = Fx_dist/m;     


	         gp_x_Xdata(0, 0) = Fx_dist_t;  
	         gp_x_Xdata(0, 1) = velocity[1];
	         gp_x_Xdata(0, 2) = velocity[0];
	         gp_x_Xdata(0, 3) = control[0];

	         gp_x_Xpred(0, 0) = Fx_dist;
	         gp_x_Xpred(0, 1) = velocity[1];
	         gp_x_Xpred(0, 2) = velocity[0];
	         gp_x_Xpred(0, 3) = control[0];



	         gp_x_Ydata(0, 0) = Fx_dist ;

             Fx_dist_t = Fx_dist;  //disturbance at t-1

             gp_x_.add_sample(gp_x_Ydata(0, 0), gp_x_Xdata);
             
             

             std::tie(mu_x, cov_x) = gp_x_.predict(gp_x_Xpred);


             //std::cout << "gp Y: " <<  gp_x_Ydata(0, 0);

             //std::cout << "gp X: " <<  mu_x(0, 0);

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



            Fy_dist = (m - Y_vd) * acceleration[1] -
            (control[1] -
            (m * velocity[0] - X_ud * velocity[0]) * -velocity[5] +
            (Y_v + Y_vc * sqrt(velocity[1] * velocity[1])+0.000001) * velocity[1]);

            Fy_dist = Fy_dist/m; 


	         gp_y_Xdata(0, 0) = Fy_dist_t;
	         gp_y_Xdata(0, 1) = velocity[1];
	         gp_y_Xdata(0, 2) = velocity[0];
	         gp_y_Xdata(0, 3) = control[0];

	         gp_y_Xpred(0, 0) = Fy_dist;
	         gp_y_Xpred(0, 1) = velocity[1];
	         gp_y_Xpred(0, 2) = velocity[0];
	         gp_y_Xpred(0, 3) = control[0];



	         gp_y_Ydata(0, 0) = Fy_dist;

             Fy_dist_t = Fy_dist;  //disturbance at t-1



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


	         gp_z_Xdata(0, 0) = velocity[1];
	         gp_z_Xdata(0, 1) = velocity[1];
	         gp_z_Xdata(0, 2) = velocity[1];
	         gp_z_Xdata(0, 3) = velocity[1];

	         gp_z_Xpred(0, 0) = velocity[1];
	         gp_z_Xpred(0, 1) = velocity[1];
	         gp_z_Xpred(0, 2) = velocity[1];
	         gp_z_Xpred(0, 3) = velocity[1];




	         gp_z_Ydata(0, 0) = acceleration.at(2) ;

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
        std_msgs::msg::Float64MultiArray msgx;
        std_msgs::msg::Float64MultiArray msgy;
        std_msgs::msg::Float64MultiArray msgz;
        msg.data = {mu_x(0, 0), mu_y(0, 0), mu_z(0, 0)};
      for (int i = 0; i < 30; ++i) {
        msgx.data.push_back(mu_x(0, 0));
        msgy.data.push_back(mu_y(0, 0));
        msgz.data.push_back(mu_z(0, 0));
    }

        
        gp_pred_mu_publisher_->publish(msg);

        gp_pred_mu_x_publisher_->publish(msgx);
        gp_pred_mu_y_publisher_->publish(msgy);
        gp_pred_mu_z_publisher_->publish(msgz);

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
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr gp_pred_mu_x_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr gp_pred_mu_y_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr gp_pred_mu_z_publisher_;


    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr gp_pred_cov_publisher_;


    GaussianProcess gp_x_;
    GaussianProcess gp_y_;
    GaussianProcess gp_z_;
    

};






void GP::mpc_cb(const geometry_msgs::msg::Wrench::SharedPtr msg) {

    control = {  msg->force.x,
    	              msg->force.y, 
                      msg->force.z,
                      msg->torque.z};
}



void GP::accel_cb(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {
    acceleration = {msg->vector.x, msg->vector.y, msg->vector.z};



}

void GP::odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg) {  
        // Extract data from Odometry message
    position = {msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z};
    velocity = {msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z,
                msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z};
    }








int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
     auto gp_node = std::make_shared<GP>();


     std::thread threadGPX(&GP::predictGPX, gp_node);
     std::thread threadGPY(&GP::predictGPY, gp_node);
     std::thread threadGPZ(&GP::predictGPZ, gp_node);


        while (rclcpp::ok()){ 


        //while (gp_node->prediction_counter_ < 3) {
		//    std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Adjust as needed

		//}
        
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