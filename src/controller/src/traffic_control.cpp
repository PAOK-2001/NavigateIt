#include <ros/ros.h>
#include <ros/console.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <cmath>
#include <chrono>
#include <string>
#include <map>

using namespace std;

int nodeRate = 100;
string old_last_color, last_color = "traffic-red";
float err, out;
bool is_receiving = false;

//######################TEMPORAL
float velocity = 0.1232;
//##############################


map<string, float> traffic_multiplier_map= { 
  {"traffic-green", 1},
  {"traffic-red", 0},
  {"traffic-yellow", 0.5}
};

void receive_traffic_light(const std_msgs::String &received_color){
    last_color = received_color.data;
}

void receive_line_error(const std_msgs::Float32 &err_msg){
    err = err_msg.data;
    is_receiving = true;
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "controller");
    ros::NodeHandle handler;
    ros::Subscriber lane_error = handler.subscribe("/line_error", 10, receive_line_error); 
    ros::Subscriber traffic_light = handler.subscribe("/traffic_light", 10, receive_traffic_light);
    ros::Publisher controller_output = handler.advertise<geometry_msgs::Twist>("/cmd_vel", 10);

    float kpr, kir, kdr;
    float dt = 0;
    ros::param::get("/kpr", kpr);
    ros::param::get("/kir", kir);
    ros::param::get("/kdr", kdr);
    float rate_err, dTime;
    float last_err = 0.0;
    float cum_err = 0.0;
    ros::Rate rate(nodeRate);
    geometry_msgs::Twist output;   
    float angular_max = 1.75;
    chrono::steady_clock::time_point time = chrono::steady_clock::now();
    while(ros::ok){
        dt = (chrono::duration_cast<chrono::microseconds> (chrono::steady_clock::now() - time).count())/1000000.0;
        if(dt > 0 && is_receiving) {
            time = chrono::steady_clock::now();
            cum_err += err*dt;
            if(abs(cum_err) > abs(angular_max)/16){
                cum_err = angular_max/16 * abs(cum_err)/cum_err;
            }
            rate_err = (err - last_err) / dt;
            last_err = err;
            out = (kpr*err)+(kdr*rate_err)+(kir*cum_err);
        } else {
            out = 0;
        }

        out = angular_max* tanh(out);

        output.linear.x = velocity;
        output.angular.z = out;

        old_last_color = last_color;
        controller_output.publish(output);

        ros::spinOnce();
        rate.sleep();
    }
 
}