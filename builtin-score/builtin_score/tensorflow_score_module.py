import tensorflow as tf
import numpy as np
import os
import pandas as pd

class TensorflowScoreModule(object):

    def __init__(self, model_meta_path):
        model_path = os.path.dirname(model_meta_path)
        self.class_names = ["class:{}".format(str(i)) for i in range(10)]
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(model_meta_path)
        saver.restore(self.sess,tf.train.latest_checkpoint(model_path))

        graph = tf.get_default_graph()
        # TODO: fix the hard-code x,y here
        self.x = {"x": graph.get_tensor_by_name("x:0")}

        y = graph.get_tensor_by_name("y:0")
        y_label = graph.get_tensor_by_name("y_label:0") # TODO: understand why :0
        self.y = [y, y_label]
        self.y_names = ["y", "y_label"]
        print(f"Successfully loaded model from {model_path}")

    def run(self, df):
        predictions = self.sess.run(self.y, feed_dict= self.feed_dict(df))
        resultdf = pd.DataFrame()
        for index in range(len(self.y_names)):
            name = self.y_names[index]
            #TODO : fix the hard-code float64
            predict = predictions[index].astype(np.float64).tolist()
            resultdf.insert(len(resultdf.columns), name, predict, True)

        return resultdf

    def feed_dict(self, df):
        dict = {}
        for name, tensor in self.x.items():
            dict[tensor] = df[name].values
        return dict
