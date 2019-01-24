from styx_msgs.msg import TrafficLight
import os
import numpy as np
import tensorflow as tf
import time
import cv2
import datetime

class TLClassifier(object):
    def __init__(self, is_site):
        # TODO load classifier
        path = 'light_classification/model/sim_graph.pb'
        if is_site:
            path = 'light_classification/model/site_graph.pb'
            
        sess = None
        with tf.Session(graph=tf.Graph(), config=tf.ConfigProto()) as sess:
            graph_defintion = tf.GraphDef()
            with tf.gfile.Open(path, 'rb') as fid:
                data = fid.read()
                graph_defintion.ParseFromString(data)
            tf.import_graph_def(graph_defintion, name='')

        self.sess = sess
        self.graph = self.sess.graph
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.graph.get_tensor_by_name('detection_classes:0')

        self.category_index = {1: {'id': 1, 'name': u'red'}, 2: {
            'id': 2, 'name': u'yellow'}, 3: {'id': 3, 'name': u'green'}}

    def get_classification(self, image, wp = 0):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        
        input_image = np.expand_dims(cv2_image, axis=0)
        (boxes, scores, classes) = self.sess.run([self.boxes, self.scores, self.classes], 
                                                  feed_dict={self.image_tensor: input_image})

        prediction = 4
        min_score_thresh=.6
        sq_boxes = np.squeeze(boxes)
        sq_classes = np.squeeze(classes).astype(np.int32)
        sq_scores = np.squeeze(scores)

        for i in range(sq_boxes.shape[0]):
            if sq_scores is None or sq_scores[i] > min_score_thresh:
                if sq_classes[i] in self.category_index.keys():
                    prediction = sq_classes[i]
                    print("Found traffic light: {ID:%s  color:%s  pred_score:%.4f}"%(prediction, str(self.category_index[sq_classes[i]]['name']), sq_scores[i]))
                    min_score_thresh = sq_scores[i] 

        if prediction == 1:
            return TrafficLight.RED
        elif prediction == 2:
            return TrafficLight.YELLOW
        elif prediction == 3:
            return TrafficLight.GREEN
        return TrafficLight.UNKNOWN
