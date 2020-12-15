"""
This file contains all necessary auxiliary functions that might be used at 
model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from math import sqrt as sqrt
import numpy as np
import itertools
from torch.autograd import Function
import torch.nn.init as init
from ssd_project.model import ssd
from ssd_project.functions.detection import *
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#color detection of red and green color in image
#necessary libraries 
import matplotlib.patches as patches
from imutils import paths
import argparse
import imutils
import os.path
from cv2 import cv2
import re
from PIL import Image

def return_num_img(img_path):
    arr = img_path.split("/")
    name = arr[len(arr)-1]
    arr = name.split("_")
    num = arr[len(arr)-1].split(".")[0]
    return num

def compare(position_window,position_building):
    """
    If the position of window lay outside the building reture False, otherwise True
    """
    if((position_window[0]>position_building[0])&(position_window[2]<position_building[2])&(position_window[3]<position_building[3])&(position_window[1]>position_building[1])):
        return True #in the building
    return False

def green_area(image):
    green = np.uint8([[[0, 255, 0]]])  #green color
    hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV) #hsv value of green color 
    lowerLimit = hsvGreen[0][0][0] - 30, 40, 40  # range of green color lower limit and upper limit
    upperLimit = hsvGreen[0][0][0] + 10, 255, 255
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert the image into hsv
    lg = np.array(lowerLimit) #range of green color
    ug = np.array(upperLimit)
    #print(lg,ug)
    green_mask = cv2.inRange(hsv, lg, ug) #green masked image    
    contours, hiera = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    for i in contours:
        area += cv2.contourArea(i)
    return area

def get_numbers_from_filename(filename):
    return re.search(r'\d+', filename).group(0)

# combine two images into one, for pix2pix train.
def merge_images(file1, file2):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))
    return result