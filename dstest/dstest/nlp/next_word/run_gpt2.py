from dstest.nlp.next_word import sample, model
from builtin_score import ioutil
import tensorflow as tf
import numpy as np
import os
import json
import pandas as pd


from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"Write article with GPT-2")
logger = logging.getLogger(__name__)


class GPT2Runner(object):
    def __init__(self, model_path,
                 seed=None,
                 nsamples=1,
                 batch_size=1,
                 length=None,
                 temperature=1,
                 top_k=0
                 ):
        self.model_path = model_path
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0
        self.nsamples = nsamples
        self.batch_size = batch_size
        hparams = model.default_hparams()
        with open(os.path.join(model_path, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))
        self.hparams = hparams
        self.seed = seed
        self.temperature = temperature
        self.top_k = top_k
        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
        self.length = length

    def run(self, input_path):
        df = ioutil.read_parquet(input_path)
        with tf.Session(graph=tf.Graph()) as sess:
            context = tf.placeholder(tf.int32, [self.batch_size, None])
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)
            output = sample.sample_sequence(
                hparams=self.hparams, length=self.length,
                context=context,
                batch_size=self.batch_size,
                temperature=self.temperature, top_k=self.top_k
            )

            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(self.model_path)
            saver.restore(sess, ckpt)
            context_tokens = df.values.flatten()
            generated = 0
            for _ in range(self.nsamples // self.batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(self.batch_size)]
                })[:, len(context_tokens):]
                for i in range(self.batch_size):
                    generated += 1
                    result = out[i]
                    return result

@click.command()
@click.option('--model_path')
@click.option('--encoded_input_path', default="outputs/gpt2")
@click.option('--output_path', default="outputs/gpt2/output")
def run_pipeline(model_path, encoded_input_path, output_path):
    print(f'MODEL_PATH: {os.listdir(model_path)}')
    print(f'INPUT_FILES: {os.listdir(encoded_input_path)}')
    gpt2runner = GPT2Runner(model_path)
    result = gpt2runner.run(encoded_input_path)
    print(f'RESULT: {result}')
    ioutil.save_parquet(pd.DataFrame(result), output_path)
    print(f'OUTPUT_PATH: {os.listdir(output_path)}')

# python -m dstest.nlp.next_word.run_gpt2 --model_path dstest/nlp/next_word/models/117M
if __name__ == '__main__':
    run_pipeline()
