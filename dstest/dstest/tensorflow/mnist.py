from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
import tensorflow as tf
import logging
import os
from ruamel.yaml import YAML

# Test dynamic install package
from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"in dstest echo")
logger = logging.getLogger(__name__)

def save_model_spec(model_path, multiple_output):
    yaml=YAML()
    yaml.default_flow_style = False
    spec = {
        'flavor' : {
            'framework' : 'tensorflow'
        },
        "tensorflow": {
            "saved_model_path": 'deep_mnist_model.meta',
            'inputs' : [
                {
                    'name': 'x'
                }
            ],
        },
    }

    if(multiple_output == True):
        print("Write spec with multiple outputs")
        spec['tensorflow']['outputs'] = [
                {
                    'name': 'y'
                },
                {
                    'name': 'y_label'
                }
            ]
    else:
        print("Write spec with single outputs")
        spec['tensorflow']['outputs'] = [
            {
                'name': 'y'
            }
        ]

    with open(os.path.join(model_path, "model_spec.yml"), 'w') as fp:
        yaml.dump(spec, fp)

def save_model(model_path, sess):
    saver = tf.train.Saver(save_relative_paths=True)

    if(not model_path.endswith('/')):
        model_path += '/'
    
    if not os.path.exists(model_path):
        logger.info(f"{model_path} not exists")
        os.makedirs(model_path)
    else:
        logger.info(f"{model_path} exists")

    saver.save(sess, model_path + "deep_mnist_model")

@click.command()
@click.option('--action', default="train", 
        type=click.Choice(['predict', 'train']))
@click.option('--model_path', default="./model/")
@click.option('--multiple_output', default=True)
def run_pipeline(
    action, 
    model_path,
    multiple_output
    ):
    x = tf.placeholder(tf.float32, [None,784], name="x")
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x,W) + b, name="y")
    y_ = tf.placeholder(tf.float32, [None, 10])
    y_label = tf.argmax(y, 1, name= "y_label")

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    tf.summary.scalar('cross_entropy', cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(model_path + '/train', sess.graph)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
        train_writer.add_summary(summary, i)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_:mnist.test.labels}))
    train_writer.close()

    save_model(model_path, sess)
    save_model_spec(model_path, multiple_output)
    logger.info(f"training finished")

# python -m dstest.tensorflow.mnist  --model_path model/tensorflow-minist --multiple_output False
if __name__ == '__main__':
    run_pipeline()
    
