import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os, sys
import scipy.ndimage
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
from skimage import measure, morphology
import random

#parameters
np.random.seed(314159)
random.seed(314159)


path = '/media/sohn/Storage/mdata/CBIS-DDSM/'

#Loading Data
def file_names(path):
    x=[]
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            x.append(os.path.join(root, name))
    return x

def load_dicom(pathlist):
    y = []
    for i in pathlist:
        y.append(dicom.read_file(i))
    return y

def get_pixels_hu(scans):
    image = [s.pixel_array for s in scans]
    #image = image.astype(np.int16)
    return image