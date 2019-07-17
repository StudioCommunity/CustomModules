import tensorflow as tf
import numpy as np
import pandas as pd
from builtin_score.builtin_score_module import *
from builtin_score.tensorflow_score_module import *

model_path = "model/tensorflow-minist/"

def test_tensor(df):
    with open(model_path + "model_spec.yml") as fp:
        config = yaml.safe_load(fp)

    tfmodule = TensorflowScoreModule(model_path, config)
    result = tfmodule.run(df)
    print(result)

def test_builtin(df):
    module = BuiltinScoreModule(model_path)
    result = module.run(df)
    print(result)

def prepare_input():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    df = pd.DataFrame(data=batch_xs, columns=['x']*784, dtype=np.float64)

    names = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
    data = [[7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8]]
    df1 = pd.DataFrame(data=data, columns=names)
    
    df = pd.concat([df, df1], axis=1)    
    #df.to_csv("test.csv")
    return df

# python -m dstest.tensorflow.mnist_test
if __name__ == '__main__':
    df = prepare_input()
    test_builtin(df)