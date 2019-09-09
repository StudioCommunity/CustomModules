from dstest.nlp.next_word.encode import BPEEncoder
from builtin_score import ioutil
import pandas as pd
import os

from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"Decode text with BPE")
logger = logging.getLogger(__name__)

DICT_PATH_KEY = "DictionaryPath"
VOCAB_PATH_KEY = "VocabularyPath"


@click.command()
@click.option('--dict_path')
@click.option('--vocab_path')
@click.option('--encoded_token_path', default="outputs/gpt2/output")
@click.option('--decoded_text_path', default="outputs/gpt2/generated")
def run_pipeline(dict_path, vocab_path, encoded_token_path, decoded_text_path):
    print(f'ENCODED_TOKENS_PATH: {os.listdir(encoded_token_path)}')
    df = ioutil.read_parquet(encoded_token_path)
    meta = {
        DICT_PATH_KEY: dict_path,
        VOCAB_PATH_KEY: vocab_path
    }
    encoder = BPEEncoder(meta)
    result = encoder.decode(df)
    ioutil.save_parquet(result, decoded_text_path)


# python -m dstest.nlp.next_word.decode --dict_path https://wanhanamlservi5915456327.blob.core.windows.net/gpt2/encoder.json?sp=r&st=2019-08-20T09:54:41Z&se=2019-08-31T17:54:41Z&spr=https&sv=2018-03-28&sig=y%2Bipiu7KlerUzPNKNlOyq7dsLAWQnuPEpX2N3D86oSc%3D&sr=b --vocab_path https://wanhanamlservi5915456327.blob.core.windows.net/gpt2/vocab.bpe?sp=r&st=2019-08-20T09:55:16Z&se=2019-08-31T17:55:16Z&spr=https&sv=2018-03-28&sig=q5F4n%2BE3F0CKjo1kiD5RkZ8v6tzfDAnLX0xYOb3umtk%3D&sr=b
if __name__ == '__main__':
    run_pipeline()
