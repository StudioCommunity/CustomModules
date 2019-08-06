import os
import logging
import click
import pandas as pd
from builtin_score import ioutil
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"in {__file__} v1")
logger = logging.getLogger(__name__)

def get_category(prob, categories):
  pred = np.argsort(prob)[::-1] ##[::-1] inverse order
  # Get top1 label
  top1 = categories[pred[0]]
  # Get top5 label
  #top5 = [synset[pred[i]] for i in range(5)]
  return top1

def get_categories(prob, categories):
  result=[]
  for i in range (len(prob)): 
    category = get_category(prob[i], categories)
    result.append(category)
  df = pd.DataFrame({'Category': result})
  return df

class Process:
  def __init__(self, meta: dict = {}):
    self.prob_col = str(meta.get('Probability Column Name', ''))
    self.file_name = str(meta.get('Category File Name', ''))
    logger.info(f"reading from {self.file_name}")
    self.categories = [l.strip() for l in open(self.file_name).readlines()]

  def run(self, input_df: pd.DataFrame, meta: dict = None):
    print(input_df.columns)
    return get_categories(input_df[self.prob_col], self.categories)

@click.command()
@click.option('--input_path', default="datas/mnist")
@click.option('--meta_path', default="model/vgg")
@click.option('--output_path', default="outputs/mnist")
@click.option('--file_name', default="")
@click.option('--prob_col', default="")
def run(input_path, meta_path, output_path, file_name, prob_col):
  """
  read
  """
  
  meta = {
    "Category File Name": os.path.join(meta_path, file_name),
    "Probability Column Name": prob_col
  }

  proccesor = Process(meta)
  df = ioutil.read_parquet(input_path)
  result = proccesor.run(df)
  ioutil.save_parquet(result, output_path,True)

# python -m dstest.postprocess.prob_to_category  --input_path outputs/imagenet/ouput --meta_path model/vgg --output_path outputs/imagenet/categories --file_name=synset.txt --prob_col=import/prob
if __name__ == '__main__':
  run()
