import onnx
import onnxruntime as ort
import numpy as np
import pandas as pd
import sys
import pickle
import os

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class OnnxScoreModule(object):
    def __init__(self, model_path, config):
        pt_config = config['onnxruntime']
        self.model_file = os.path.join(model_path, pt_config['model_file_path'])
        logger.info(f'MODLE_FILE={self.model_file}')
        session = ort.InferenceSession(self.model_file)
        input_names = [entry['name'] for entry in pt_config['inputs']]
        output_names = [entry['name'] for entry in pt_config['outputs']]
        self.wrapper = OnnxWrapper(session, input_names, output_names)

    def run(self, df):
        return self.wrapper.predict(df)


class OnnxWrapper(object):
    def __init__(self, session, input_names, output_names):
        self.session = session
        self.input_names = input_names
        self.output_names = output_names

    def extract(self, name):
        return name.partition(':')[0]
    
    def predict(self, df):
        feed_dict = {}
        for name in self.input_names:
            data = df[self.extract(name)][0]
            if data.dtype == np.float64:
                data = data.astype(np.float32)
            feed_dict[name] = data.reshape(-1, data.shape[0])

        output = self.session.run(self.output_names, input_feed=feed_dict)
        data = {}
        for idx, name in enumerate(self.output_names):
            data[self.extract(name)] = output[idx].tolist()
        output_df = pd.DataFrame(data)
        return output_df
