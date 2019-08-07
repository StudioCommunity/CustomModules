import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from builtin_score.builtin_score_module import *
from builtin_score.tensorflow_score_module import *
from builtin_score import ioutil

model_path = "model/tensorflow-minist/"

def test_tensor(df):
    with open(model_path + "model_spec.yml") as fp:
        config = yaml.safe_load(fp)

    tfmodule = TensorflowScoreModule(model_path, config)
    schema = tfmodule.get_schema()
    print('#################')
    print(schema)
    with open(os.path.join(model_path, 'contract.json'), 'w') as f:
        json.dump(schema, f)
    
    result = tfmodule.run(df)
    print(result)

def test_builtin(df):
    module = BuiltinScoreModule(model_path, {"Append score columns to output": "True"})
    result = module.run(df)
    print(result)
    return result

def prepare_input():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    print(batch_xs)
    #df = pd.DataFrame()
    #df.insert(len(df.columns), 'x', batch_xs.tolist(), True)
    
    columns = [f"x.{i}" for i in range(784)]
    #columns = ['x']*784
    df = pd.DataFrame(data=batch_xs, columns=columns, dtype=np.float64)

    names = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
    data = [[7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8]]
    df1 = pd.DataFrame(data=data, columns=names)
    
    df = pd.concat([df, df1], axis=1)    
    #df.to_parquet("test.parquet")
    #df.to_csv("test.csv")
    return df

# python -m dstest.tensorflow.mnist_test
if __name__ == '__main__':
    # df = prepare_input()
    df = ioutil.read_parquet("../dstest/outputs/mnist/")
    print(df.columns)

    test_tensor(df)
    out = test_builtin(df)
    print(out.columns)
    print(out)