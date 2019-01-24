from styx_msgs.msg import TrafficLight
import os
import tensorflow as tf
import numpy as np
import cv2
import rospy
import datetime

class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier

        path = r'light_classification/model/sim_frozen_graph.pb'
        if is_site:
            path = r'light_classification/model/site_frozen_graph.pb'
            
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_defintion = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                graph_defintion.ParseFromString(fid.read())
                tf.import_graph_def(graph_defintion, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

            self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        with self.graph.as_default():
            input_image = np.expand_dims(image, axis=0)
            (boxes, scores, classes,num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: input_image})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        print("Found traffic light: {color:%s  pred_score:%.4f}"%(classes[0], scores[0]))
        if scores[0] > .5:
            if classes[0] == 1:
                return TrafficLight.GREEN
            elif classes[0] == 2:
                return TrafficLight.RED
            elif classes[0] == 3:
                return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN