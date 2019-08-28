import json
import regex as re
import pandas as pd
import urllib.request
import os
from functools import lru_cache
from builtin_score import ioutil


from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"Encode text with BPE")
logger = logging.getLogger(__name__)

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BPEEncoder(object):
    def __init__(self, dict_path, vocab_path, errors='replace'):
        self.byte_encoder = bytes_to_unicode()
        response = urllib.request.urlopen(dict_path)
        self.dict = json.load(response)
        response = urllib.request.urlopen(vocab_path)
        bpe_vocab = response.read().decode('utf-8')
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_vocab.split('\n')[1:-1]]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.decoder = {v:k for k,v in self.dict.items()}
        self.errors = errors  # how to handle errors in decoding

    def encode(self, raw_text):
        bpe_tokens = []
        for token in re.findall(self.pat, raw_text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.dict[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def bpe(self, token):
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        return word

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

@click.command()
@click.option('--dict_path')
@click.option('--vocab_path')
@click.option('--raw_text')
@click.option('--output_path', default="outputs/gpt2")
def run_pipeline(dict_path, vocab_path, raw_text, output_path):
    encoder = BPEEncoder(dict_path, vocab_path)
    result = encoder.encode(raw_text)
    print(f'result: {result}')
    df = pd.DataFrame()
    df["input:0"] = result
    ioutil.save_parquet(df, output_path)
    print(f'Output path: {os.listdir(output_path)}')


# python -m dstest.nlp.next_word.encode.py --dict_path https://wanhanamlservi5915456327.blob.core.windows.net/gpt2/encoder.json?sp=r&st=2019-08-19T06:20:30Z&se=2019-08-19T14:20:30Z&spr=https&sv=2018-03-28&sig=qTGkAtWTCd53nk422%2BdBu2kNOfzlgQP4kPQ5UBJwHdE%3D&sr=b --vocab_path https://wanhanamlservi5915456327.blob.core.windows.net/gpt2/vocab.bpe?sp=r&st=2019-08-19T06:20:07Z&se=2019-08-19T14:20:07Z&spr=https&sv=2018-03-28&sig=ADyXdQpwS4eewX5q2w3ab7FnQoR6SGKuE6yKKcifi5M%3D&sr=b --raw_text "This is a test"
if __name__ == '__main__':
    run_pipeline()
