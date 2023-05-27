#include <ros/ros.h>
#include <std_msgs/String.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>

class LightDetector {
    private:
        ros::NodeHandle nh;
        ros::Publisher light_publisher;
        cv_bridge::CvBridge* img_bridge;
        ros::Rate* rate;
        ros::Subscriber img_subscriber;
        cv::Mat dash_cam;
        int _input_width;
        int _input_height;
        std::vector<double> _threshold;
        std::vector<std::string> _classes;
        std::vector<cv::Scalar> _display_colors;
        cv::dnn::Net classifier;
        int prev_color;
    public:
        LightDetector(bool use_cuda) {
            light_publisher = nh.advertise<std_msgs::String>("/traffic_light", 1);
            img_bridge = new cv_bridge::CvBridge();
            rate = new ros::Rate(15);
            img_subscriber = nh.subscribe("/video_source/dash_cam", 1, &LightDetector::receive_img, this);
            _input_width = 640;
            _input_height = 640;
            _threshold = {0.2, 0.4, 0.4}; // score, nms, confidence
            _classes = {"changing-gy", "changing-rg", "changing-ry", "forward", "line", "stop", "traffic-green", "traffic-red", "traffic-yellow", "turn-right"};
            _display_colors = {cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255), cv::Scalar(100, 100, 100), cv::Scalar(0, 0, 128), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
            classifier = cv::dnn::readNetFromONNX("/home/puzzlebot/traffic-v2.onnx");
            prev_color = 3;
            if (use_cuda) {
                std::cout << "Running classifier on CUDA" << std::endl;
                classifier.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                classifier.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
            } else {
                std::cout << "Running classifier on CPU" << std::endl;
                classifier.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                classifier.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
        }

        void receive_img(const sensor_msgs::Image::ConstPtr& img) {
            cv::Mat cam = img_bridge->imgmsg_to_cv2(img->image);
            dash_cam = cam.clone();
        }

        cv::Mat format_frame(const cv::Mat& frame) {
            int row = frame.rows;
            int col = frame.cols;
            int _max = std::max(col, row);
            cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
            frame.copyTo(result(cv::Rect(0, 0, col, row)));
            return result;
        }

        cv::Mat detect_lights(const cv::Mat& frame) {
            cv::Mat blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(_input_width, _input_height), true, false);
            classifier.setInput(blob);
            cv::Mat preds = classifier.forward();
            return preds;
        }

        void wrap_detection(const cv::Mat& input_image, const cv::Mat& output_data, std::vector<int>& class_ids, std::vector<float>& confidences, std::vector<cv::Rect>& boxes) {
            int rows = output_data.rows;
            int image_width = input_image.cols;
            int image_height = input_image.rows;

            double x_factor = image_width / static_cast<double>(_input_width);
            double y_factor = image_height / static_cast<double>(_input_height);

            for (int r = 0; r < rows; ++r) {
                const float* row = output_data.ptr<float>(r);
                float confidence = row[4];

                if (confidence >= 0.4) {
                    const float* classes_scores = row + 5;
                    cv::Point max_indx;
                    cv::minMaxLoc(cv::Mat(1, static_cast<int>(_classes.size()), CV_32F, classes_scores), nullptr, nullptr, nullptr, &max_indx);
                    int class_id = max_indx.x;

                    if (classes_scores[class_id] > 0.25) {
                        confidences.push_back(confidence);
                        class_ids.push_back(class_id);

                        std_msgs::String msg;
                        msg.data = _classes[class_id];
                        if (prev_color != class_id) {
                            light_publisher.publish(msg);
                            prev_color = class_id;
                        }

                        float x = row[0];
                        float y = row[1];
                        float w = row[2];
                        float h = row[3];

                        int left = static_cast<int>((x - 0.5 * w) * x_factor);
                        int top = static_cast<int>((y - 0.5 * h) * y_factor);
                        int width = static_cast<int>(w * x_factor);
                        int height = static_cast<int>(h * y_factor);

                        boxes.push_back(cv::Rect(left, top, width, height));
                    }
                }
            }

            std::vector<int> indexes;
            cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);

            std::vector<int> result_class_ids;
            std::vector<float> result_confidences;
            std::vector<cv::Rect> result_boxes;

            for (int i : indexes) {
                result_class_ids.push_back(class_ids[i]);
                result_confidences.push_back(confidences[i]);
                result_boxes.push_back(boxes[i]);
            }

            class_ids = result_class_ids;
            confidences = result_confidences;
            boxes = result_boxes;
        }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "traffic_detector");
    LightDetector detector(true);

    while (ros::ok()) {
        cv::Mat frame = detector.dash_cam;
        cv::Mat frame_yolo = detector.format_frame(frame);
        cv::Mat preds = detector.detect_lights(frame_yolo);

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        detector.wrap_detection(frame_yolo, preds, class_ids, confidences, boxes);

        for (size_t i = 0; i < class_ids.size(); ++i) {
            int class_id = class_ids[i];
            float confidence = confidences[i];
            cv::Rect box = boxes[i];

            cv::Scalar color = detector._display_colors[class_id % detector._display_colors.size()];

            cv::rectangle(frame, box, color, 2);
            cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, -1);
            cv::putText(frame, detector._classes[class_id], cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }

        cv::imshow("output", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }

        detector.rate->sleep();
        ros::spinOnce();
    }

    cv::destroyAllWindows();
    return 0;
}
