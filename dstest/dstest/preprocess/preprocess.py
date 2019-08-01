import logging
import click
import pandas as pd
from builtin_score import ioutil
import math
from scipy import ndimage
import numpy as np
import base64
import cv2
from . import datauri_util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"in {__file__} v1")
logger = logging.getLogger(__name__)

def getBestShift(img):
  cy,cx = ndimage.measurements.center_of_mass(img)

  rows,cols = img.shape
  shiftx = np.round(cols/2.0-cx).astype(int)
  shifty = np.round(rows/2.0-cy).astype(int)

  return shiftx,shifty

def shift(img,sx,sy):
  rows,cols = img.shape
  M = np.float32([[1,0,sx],[0,1,sy]])
  shifted = cv2.warpAffine(img,M,(cols,rows))
  return shifted

def transform_image(base64Str, no):
  """
  transform image to 28x28x3
  """
  gray = datauri_util.readb64(base64Str)
  gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
  #cv2.imwrite("outputs/test_1_gray.png", img)

  # rescale it
  gray = cv2.resize(255-gray, (28, 28))
  
  # better black and white version
  (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  #cv2.imwrite('outputs/test_2_rescale.png',gray)

  while np.sum(gray[0]) == 0:
      gray = gray[1:]

  while np.sum(gray[:,0]) == 0:
    gray = np.delete(gray,0,1)

  while np.sum(gray[-1]) == 0:
    gray = gray[:-1]

  while np.sum(gray[:,-1]) == 0:
    gray = np.delete(gray,-1,1)

  #print(gray.shape)
  rows,cols = gray.shape

  if rows > cols:
    factor = 20.0/rows
    rows = 20
    cols = int(round(cols * factor))
    # first cols than rows
    gray = cv2.resize(gray, (cols, rows))
  else:
    factor = 20.0/cols
    cols = 20
    rows = int(round(rows * factor))
    # first cols than rows
    gray = cv2.resize(gray, (cols, rows))

  colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
  rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
  gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
  # cv2.imwrite('outputs/test_3.png',gray)

  shiftx, shifty = getBestShift(gray)
  shifted = shift(gray, shiftx, shifty)
  gray = shifted

  return gray

def add_data_to_dataframe(input_df, results, target_column):
  if(input_df.columns.contains(target_column)):
    logger.info(f"writing to column {target_column}")
    input_df[target_column] = results
  else:
    logger.info(f"append new column {target_column}")
    input_df.insert(len(input_df.columns), target_column, results, True)

class PreProcess:
  def __init__(self, meta: dict = {}):
    self.image_column = str(meta.get('Image Column', 'image'))
    self.target_column = str(meta.get('Target Column', ''))
    self.target_data_column = str(meta.get('Target DataURI Column', ''))
    
    if not self.target_column:
      self.target_column = self.image_column

    if not self.target_data_column:
      self.target_data_column = f"{self.target_column}_data"

    print(self.image_column, self.target_column)

  def run(self, input_df: pd.DataFrame, meta: dict = None):
    results = []
    datauris = []
    
    for index, row in input_df.iterrows():
      #print(row['label'])
      img = transform_image(row[self.image_column], index)
      
      # append datauris
      if self.target_data_column:
        datauri = datauri_util.img_to_datauri(img)
        datauris.append(datauri)
      
      # save the processed images
      #cv2.imwrite("outputs/image_"+str(index)+".png", gray)
      # Convert to 0-1 based range
      flatten = img.flatten() / 255.0
      results.append(flatten)

    add_data_to_dataframe(input_df, results, self.target_column)
    if self.target_data_column:
      add_data_to_dataframe(input_df, datauris, self.target_data_column)

    return input_df

@click.command()
@click.option('--input_path', default="datas/mnist")
@click.option('--output_path', default="outputs/mnist")
@click.option('--image_column', default="image")
@click.option('--target_column', default="x")
@click.option('--target_datauri_column', default="")
def run(input_path, output_path, image_column, target_column, target_datauri_column):
  """
  This functions read base64 encoded images from df. Transform to format required by model input.
  """
  meta = {
    "Image Column": image_column,
    "Target Column": target_column,
    "Target DataURI Column": target_datauri_column
  }
  proccesor = PreProcess(meta)

  df = ioutil.read_parquet(input_path)
  result = proccesor.run(df)
  ioutil.save_parquet(result, output_path)

# python -m dstest.preprocess.preprocess  --input_path datas/mnist --output_path outputs/mnist --image_column=image --target_column=x --target_datauri_column=x.data
if __name__ == '__main__':
  run()
