import tensorflow as tf
import numpy as np
import os
import pandas as pd

class TensorflowScoreModule(object):

    def __init__(self, model_meta_path, config):
        model_path = os.path.dirname(model_meta_path)
        self.class_names = ["class:{}".format(str(i)) for i in range(10)]
        self.sess = tf.Session()
        print(f"model_meta_path = {model_meta_path}, model_path = {model_path}")
        saver = tf.train.import_meta_graph(model_meta_path)
        saver.restore(self.sess,tf.train.latest_checkpoint(model_path))

        graph = tf.get_default_graph()
        
        self.x = {}
        for index in range(len(config["inputs"])):
            name = config["inputs"][index]["name"]
            self.x[name] = graph.get_tensor_by_name(name + ":0")
        
        print("loaded inputs:")
        print(self.x)

        self.y = []
        self.y_names = []
        for index in range(len(config["outputs"])):
            name = config["outputs"][index]["name"]
            # TODO: support :0 in future version. :0 means the first ouput of an op in tensorflow graph
            tensor = graph.get_tensor_by_name(name +":0")
            self.y.append(tensor)
            self.y_names.append(name)

        print("loaded outputs:")
        print(self.y)
        print(self.y_names)

        print(f"Successfully loaded model from {model_path}")

    def run(self, df):
        predictions = self.sess.run(self.y, feed_dict= self.feed_dict(df))
        resultdf = pd.DataFrame()
        for index in range(len(self.y_names)):
            name = self.y_names[index]
            predict = predictions[index].tolist()
            resultdf.insert(len(resultdf.columns), name, predict, True)

        return resultdf

    def feed_dict(self, df):
        dict = {}
        for name, tensor in self.x.items():
            dict[tensor] = df[name].values
        return dict
