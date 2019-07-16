import tensorflow as tf
import numpy as np
import os
import pandas as pd

class _TFSavedModelWrapper(object):
    """
    Wrapper class that exposes a TensorFlow model for inference via a ``predict`` function such that
    ``predict(data: pandas.DataFrame) -> pandas.DataFrame``.
    """
    def __init__(self, export_dir, tf_meta_graph_tags, tf_signature_def_key):
        tf_graph = tf.Graph()
        tf_sess = tf.Session(graph=tf_graph)

        self.tf_graph = tf_graph
        self.tf_sess = tf_sess

        with tf_graph.as_default():
          signature_def = self._load_tensorflow_saved_model(tf_sess, tf_meta_graph_tags, tf_signature_def_key, export_dir)
          
        # input keys in the signature definition correspond to input DataFrame column names
        self.input_tensor_mapping = {
                tensor_column_name: tf_graph.get_tensor_by_name(tensor_info.name)
                for tensor_column_name, tensor_info in signature_def.inputs.items()
        }
        # output keys in the signature definition correspond to output DataFrame column names
        self.output_tensors = {
                sigdef_output: tf_graph.get_tensor_by_name(tnsr_info.name)
                for sigdef_output, tnsr_info in signature_def.outputs.items()
        }

    def predict(self, df):
      with self.tf_graph.as_default():
        feed_dict = {
                self.input_tensor_mapping[tensor_column_name]: df[tensor_column_name].values
                for tensor_column_name in self.input_tensor_mapping.keys()
        }
        raw_preds = self.tf_sess.run(self.output_tensors, feed_dict=feed_dict)
        resultdf = pd.DataFrame()
        for column_name, values in raw_preds.items():
          resultdf.insert(len(resultdf.columns), column_name, values.tolist(), True)
        return resultdf

    def _load_tensorflow_saved_model(self, sess, tf_meta_graph_tags,tf_signature_def_key, export_dir):
      meta_graph_def = tf.saved_model.loader.load(sess, tf_meta_graph_tags, export_dir)
      if tf_signature_def_key not in meta_graph_def.signature_def:
        raise Exception("Could not find signature def key %s" % tf_signature_def_key)
      return meta_graph_def.signature_def[tf_signature_def_key]

class _TFSaverWrapper(object):
    def __init__(self, model_path, config):
        self.sess = tf.Session()
        tf_config = config["tensorflow"]
        print(tf_config)
        model_meta_path = os.path.join(model_path, tf_config["saved_model_path"])
        
        print(f"model_meta_path = {model_meta_path}, model_path = {model_path}")
        saver = tf.train.import_meta_graph(model_meta_path)
        saver.restore(self.sess,tf.train.latest_checkpoint(model_path))
        graph = tf.get_default_graph()
        
        self.x = {}
        for index in range(len(tf_config["inputs"])):
            name = tf_config["inputs"][index]["name"]
            self.x[name] = graph.get_tensor_by_name(name + ":0")
        
        print("loaded inputs:")
        print(self.x)

        self.y = []
        self.y_names = []
        for index in range(len(tf_config["outputs"])):
            name = tf_config["outputs"][index]["name"]
            # TODO: support :0 in future version. :0 means the first ouput of an op in tensorflow graph
            tensor = graph.get_tensor_by_name(name +":0")
            self.y.append(tensor)
            self.y_names.append(name)

        print("loaded outputs:")
        print(self.y)
        print(self.y_names)

        print(f"Successfully loaded model from {model_path}")

    def predict(self, df):
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

class TensorflowScoreModule(object):

    def __init__(self, model_path, config):
        tf_config = config["tensorflow"]
        
        if(tf_config.get("serialization_format", "saver") == "saved_model"):
            export_dir = os.path.join(model_path, tf_config["saved_model_path"])
            tf_meta_graph_tags = tf_config["meta_graph_tags"]
            tf_signature_def_key = tf_config["signature_def_key"]
            self.wrapper = _TFSavedModelWrapper(export_dir, tf_meta_graph_tags, tf_signature_def_key)
        else:
            self.wrapper = _TFSaverWrapper(model_path, config)

    def run(self, df):
        return self.wrapper.predict(df)

