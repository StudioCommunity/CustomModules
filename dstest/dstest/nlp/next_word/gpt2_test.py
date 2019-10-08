from builtin_score import ioutil
import tensorflow as tf
import pandas as pd
from builtin_score.builtin_score_module import *
tf.enable_eager_execution()
from dstest.nlp.next_word.encode import BPEEncoder

INPUT_PATH = "input/gpt2"
DICT_PATH = "https://wanhanamlservi5915456327.blob.core.windows.net/gpt2/encoder.json?sp=r&st=2019-09-02T06:16:01Z&se=2019-10-12T14:16:01Z&spr=https&sv=2018-03-28&sig=%2BpjBDD2NO75pKJ52g9QrSBGVAuLMWw1TcZ6l9ny0tjQ%3D&sr=b"
VOCAB_PATH = "https://wanhanamlservi5915456327.blob.core.windows.net/gpt2/vocab.bpe?sp=r&st=2019-09-02T06:16:35Z&se=2019-10-12T14:16:35Z&spr=https&sv=2018-03-28&sig=lv1nhl4UAVWiR%2BWkXPMxmDug6C4JefaCQtiLoHp%2FtL4%3D&sr=b"
ENCODED_INPUT_PATH = "outputs/gpt2/encoded_input"
ENCODED_TOKEN_PATH = "outputs/gpt2/output"
RESULT_TOKEN_PATH = "outputs/gpt2/result"
DECODED_TEXT_PATH = "outputs/gpt2/decoded"
MODEL_PATH = "models/gpt2"

INPUT_FILE_NAME = "data.dataset.parquet" # hard coded, to be replaced, and we presume the data is DataFrame inside parquet
DICT_PATH_KEY = "DictionaryPath"
VOCAB_PATH_KEY = "VocabularyPath"

def testBuiltin():
    df = ioutil.read_parquet(ENCODED_INPUT_PATH)
    module = BuiltinScoreModule(MODEL_PATH)
    result = module.run(df)
    # Result is a data frame
    ioutil.save_parquet(result, ENCODED_TOKEN_PATH)

def run_pipeline_test(dict_path, vocab_path, input_text_path, output_path):
    input_df = pd.read_parquet(os.path.join(input_text_path, INPUT_FILE_NAME), engine="pyarrow")
    meta = {
        DICT_PATH_KEY: dict_path,
        VOCAB_PATH_KEY: vocab_path
    }
    encoder = BPEEncoder(meta)
    df = encoder.encode(input_df)
    ioutil.save_parquet(df, output_path)
    print(f'Output path: {os.listdir(output_path)}')

def run_pipeline_decode(dict_path, vocab_path, encoded_token_path, decoded_text_path):
    print(f'ENCODED_TOKENS_PATH: {os.listdir(encoded_token_path)}')
    df = ioutil.read_parquet(encoded_token_path)
    meta = {
        DICT_PATH_KEY: dict_path,
        VOCAB_PATH_KEY: vocab_path
    }
    encoder = BPEEncoder(meta)
    result = encoder.decode(df)
    ioutil.save_parquet(result, decoded_text_path)

def testInput():
    raw_test = pd.DataFrame(["this is a test"])
    ioutil.save_parquet(raw_test, INPUT_PATH)
    run_pipeline_test(dict_path=DICT_PATH, vocab_path=VOCAB_PATH, input_text_path=INPUT_PATH, output_path=ENCODED_INPUT_PATH)

def testDecode():
    run_pipeline_decode(DICT_PATH, VOCAB_PATH, ENCODED_TOKEN_PATH, DECODED_TEXT_PATH)

if __name__ == '__main__':
    # testInput()
    # testBuiltin()
    testDecode()