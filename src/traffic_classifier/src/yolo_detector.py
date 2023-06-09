#!/usr/bin/env python

import rospy 
from std_msgs.msg import String
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np

class LightDetector:
        def __init__(self, use_cuda):
            rospy.init_node("traffic_monitor")
            self.light_publisher = rospy.Publisher("/traffic_light", String, queue_size=1)
            self.img_bridge = CvBridge()
            self.rate = rospy.Rate(15)
            rospy.Subscriber("/video_source/dash_cam", Image, self.receive_img)
            self._input_width = 640
            self._input_height = 640
            self._threshold = [0.2,0.4,0.4] # score, nms, confidence
            self._classes = ['construction', 'forward', 'give_way', 'green', 'left', 'red', 'right', 'roundabout', 'stop', 'yellow']                                                                                  
            self._display_colors = [(128, 255, 128), (255, 128, 255), (85, 170, 255), (255, 255, 255),(100,100,100),(0,0,128),(0,255,0),(0,0,255),(0,255,255),(255,0,0)]
            self.classifier = cv2.dnn.readNetFromONNX("/home/puzzlebot/traffic-v2.onnx")
            self.prev_color = 3
            if use_cuda:
                print("Running classifier on CUDA")
                self.classifier.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.classifier.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            else:
                print("Running classifier on CPU")
                self.classifier.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.classifier.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
        def receive_img(self, img):
            cam = self.img_bridge.imgmsg_to_cv2(img) 
            self.dash_cam = cam.copy()

        def format_frame(self, frame):
            row, col, _ = frame.shape
            _max = max(col, row)
            result = np.zeros((_max, _max, 3), np.uint8)
            result[0:row, 0:col] = frame
            return result
        
        def detect_lights(self, frame):
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self._input_width, self._input_height), swapRB=True, crop=False)
            self.classifier.setInput(blob)
            preds = self.classifier.forward()
            return preds
        
        def wrap_detection(self, input_image, output_data):
            class_ids = []
            confidences = []
            boxes = []
            
            rows = output_data.shape[0]
            image_width, image_height, _ = input_image.shape

            x_factor = image_width / self._input_width
            y_factor =  image_height / self._input_height

            for r in range(rows):
                row = output_data[r]
                confidence = row[4]
                if confidence >= 0.4:
                    classes_scores = row[5:]
                    _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                    class_id = max_indx[1]
                    if (classes_scores[class_id] > .25):

                        confidences.append(confidence)

                        class_ids.append(class_id)
                        current_detection = self._classes[class_ids[0]]
                        if(self.prev_color != class_ids[0]):
                            self.light_publisher.publish(current_detection)
                            self.prev_color = class_ids[0]

                        x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                        left = int((x - 0.5 * w) * x_factor)
                        top = int((y - 0.5 * h) * y_factor)
                        width = int(w * x_factor)
                        height = int(h * y_factor)
                        box = np.array([left, top, width, height])
                        boxes.append(box)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

            result_class_ids = []
            result_confidences = []
            result_boxes = []

            for i in indexes:
                result_confidences.append(confidences[i])
                result_class_ids.append(class_ids[i])
                result_boxes.append(boxes[i])

            return result_class_ids, result_confidences, result_boxes
        

if __name__ == "__main__":
    detector = LightDetector(True)
    while (not rospy.is_shutdown()): 
        frame= detector.dash_cam
        frame_yolo = detector.format_frame(frame)
        preds = detector.detect_lights(frame_yolo)
        class_ids, confidences, boxes = detector.wrap_detection(frame_yolo, preds[0])
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = detector._display_colors[int(classid) % len(detector._display_colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, detector._classes[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))
        cv2.imshow("output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        detector.rate.sleep()

    cv2.destroyAllWindows()
