from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np

# This is a placeholder for a Google-internal import.
import tensorflow as tf
from builtin_score.builtin_score_module import *
from builtin_score.keras_score_module import KerasScoreModule
from keras.models import load_model
from keras.datasets import mnist


def load_model_then_predict1(model_spec_file = 'model/keras-mnist/model_spec.yml'):
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_test = x_test.reshape(x_test.shape[0], -1) # x_test shape [x_test.shape[0], 784]
  x_test = x_test[:8] # only pick 8 imgs
  x_test = x_test.astype('float32') / 255

  df = pd.DataFrame(data=x_test, columns=['img']*784, dtype=np.float64)
  df.to_csv('mnist_kera_test_data.csv')

  with open(model_spec_file) as fp:
      config = yaml.safe_load(fp)

  model_path = "./model/keras-mnist/"
  keras_model = KerasScoreModule(model_path, config)
  result = keras_model.run(df)
  print(result)

  module = BuiltinScoreModule(model_path)
  result = module.run(df)
  print('=====buildinScoreModule=======')
  print(result)

# python -m dstest.keras.saved_model_predict_test
if __name__ == '__main__':
  load_model_then_predict1()
  print('\n===============another load model method================\n')
  load_model_then_predict1('model/keras-mnist/model_weights_spec.yml')
