#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <rocon_std_msgs/StringArray.h>
#include <cmath>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

using namespace std;

int nodeRate = 25;
vector<string> old_last_signs, last_signs = {"traffic_red"};
vector<string> dirs = {"forward","left","right"};
bool intersection_likely = false;
bool is_receiving = false;
bool is_detecting = false;
float velocity = 0.0616;
float err;


unordered_map<string, float> traffic_multiplier_map= { 
  {"green", 1},
  {"yellow", 0.5},
  {"construction", 0.5},
  {"red", 0},
  {"stop", 0}
};

void receive_traffic_sign(const rocon_std_msgs::StringArray &received_signs){
    last_signs = received_signs.strings;
    is_detecting = true;
}

void receive_line_error(const std_msgs::Float32 &err_msg){
    err = err_msg.data;
    is_receiving = true;
}

void receive_intersection_likely(const std_msgs::Int32 &int_msg) {
    intersection_likely = int_msg.data >= 15;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "controller");
    ros::NodeHandle handler;
    ros::Subscriber lane_error = handler.subscribe("/line_error", 10, receive_line_error); 
    ros::Subscriber traffic_sign = handler.subscribe("/traffic_sign", 10, receive_traffic_sign);
    ros::Subscriber detect_intersection = handler.subscribe("/intersection_likely", 10, receive_intersection_likely);
    ros::Publisher controller_output = handler.advertise<geometry_msgs::Twist>("/cmd_vel", 10);

    float kpr, kir, kdr, out;
    float dt = 0;
    ros::param::get("/kpr", kpr);
    ros::param::get("/kir", kir);
    ros::param::get("/kdr", kdr);
    bool speed_sign_detected = false;
    float rate_err, dTime;
    float last_err = 0.0;
    float cum_err = 0.0;
    float speed_multiplier = 1;
    string last_dir = "forward";
    ros::Rate rate(nodeRate);
    geometry_msgs::Twist output;   
    float angular_max = 1.12;
    chrono::steady_clock::time_point time = chrono::steady_clock::now();
    while(ros::ok){
        dt = (chrono::duration_cast<chrono::microseconds> (chrono::steady_clock::now() - time).count())/1000000.0;
        if(dt > 0 && is_receiving && is_detecting) {

            // keep track of last direction sign detected (forward, right, left)
            speed_sign_detected = false;
            for (string sign : last_signs) {
                if (sign == "forward" || sign == "right" || sign == "left") {
                    last_dir = sign;
                } else {
                    auto it = traffic_multiplier_map.find(sign);
                    if (it != traffic_multiplier_map.end()) {
                        speed_multiplier = traffic_multiplier_map[sign];
                        speed_sign_detected = true;
                    }
                }
            }
            if (!speed_sign_detected && speed_multiplier != 0) speed_multiplier = 1;

            if (intersection_likely) {
                //car movement without lane
                if (last_dir == "left") {
                    out = angular_max / 3;
                } else if (last_dir == "right") {
                    out = -angular_max / 3;
                } else {
                    out = 0;
                }
            } else {
                time = chrono::steady_clock::now();
                cum_err += err*dt;
                if(abs(cum_err) > abs(angular_max)/16){
                    cum_err = angular_max/16 * abs(cum_err)/cum_err;
                }
                rate_err = (err - last_err) / dt;
                last_err = err;
                out = (kpr*err)+(kdr*rate_err)+(kir*cum_err);
            }

        } else {
            out = 0;
        }

        out = angular_max* tanh(out);

        output.linear.x = velocity*speed_multiplier;
        output.angular.z = out*speed_multiplier;

        old_last_signs = last_signs;
        controller_output.publish(output);

        ros::spinOnce();
        rate.sleep();
    }
 
}