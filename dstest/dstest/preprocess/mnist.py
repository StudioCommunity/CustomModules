import cv2
from scipy import ndimage
import numpy as np
import math

def getBestShift(img):
  cy,cx = ndimage.measurements.center_of_mass(img)

  rows,cols = img.shape
  shiftx = np.round(cols/2.0-cx).astype(int)
  shifty = np.round(rows/2.0-cy).astype(int)

  return shiftx,shifty

def shift(img, sx, sy):
  rows, cols = img.shape
  M = np.float32([[1, 0, sx], [0, 1, sy]])
  shifted = cv2.warpAffine(img, M, (cols, rows))
  return shifted

def transform_image_mnist(gray, target_size = (28, 28)):
  """
  transform image to 28x28x3
  """
  gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
  #cv2.imwrite("outputs/test_1_gray.png", img)

  # rescale it
  gray = cv2.resize(255-gray, target_size)
  
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