import logging
import click
import pandas as pd
from builtin_score import ioutil
import numpy as np
import math
from scipy import ndimage
import base64
import cv2

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"in {__file__}")
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

def readb64(base64_string):
  imgData = base64.b64decode(base64_string)
  nparr = np.fromstring(imgData, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  #print(img.shape)
  #cv2.imwrite("outputs/test_0_origin.png", img)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  #cv2.imwrite("outputs/test_1_gray.png", img)
  return img

def transform_image(base64Str, no):
  """
  transform image to 28x28x3
  """
  gray = readb64(base64Str)

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

  # save the processed images
  # cv2.imwrite("outputs/image_"+str(no)+".png", gray)
  """
  all images in the training set have an range from 0-1
  and not from 0-255 so we divide our flatten images
  (a one dimensional vector with our 784 pixels)
  to use the same 0-1 based range
  """
  flatten = gray.flatten() / 255.0
  
  return flatten

class PreProcess:
  def __init__(self, meta: dict = {}):
    self.image_column = str(meta.get('Image Column', 'image'))
    self.target_column = str(meta.get('Target Column', ''))
    if not self.target_column:
      self.target_column = self.image_column

    print(self.image_column, self.target_column)

  def run(self, input_df: pd.DataFrame, meta: dict = None):
    results = []
    
    for index, row in input_df.iterrows():
      #print(row['label'])
      img = transform_image(row[self.image_column], index)
      results.append(img)
    
    if(input_df.columns.contains(self.target_column)):
      logger.info(f"writing to column {self.target_column}")
      input_df[self.target_column] = results
    else:
      logger.info(f"append new column {self.target_column}")
      input_df.insert(len(input_df.columns), self.target_column, results, True)
    return input_df

@click.command()
@click.option('--input_path', default="datas/mnist")
@click.option('--output_path', default="outputs/mnist")
@click.option('--image_column', default="image")
@click.option('--target_column', default="x")
def run(input_path, output_path, image_column, target_column):
  """
  This functions read base64 encoded images from df. Transform to format required by model input.
  """
  meta = {
    "Image Column": image_column,
    "Target Column": target_column
  }
  proccesor = PreProcess(meta)

  df = ioutil.read_parquet(input_path)
  result = proccesor.run(df)
  ioutil.save_parquet(result, output_path)

# python -m dstest.preprocess.preprocess  --input_path datas/mnist --output_path outputs/mnist --target_column x
if __name__ == '__main__':
  run()
