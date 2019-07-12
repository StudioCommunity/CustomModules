import tensorflow as tf
import numpy as np
import pandas as pd
from buildin_score.builtin_score_module import *

# python -m dstest.tensorflow.mnist_test
if __name__ == '__main__':
    
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    batch_xs, batch_ys = mnist.train.next_batch(2)
    
    df = pd.DataFrame(data=batch_xs)
    module = BuiltinScoreModule("model/tensorflow-minist/")
    result = module.run(df)
    print(result)