import os
import pandas as pd
import numpy as np
from PIL import Image

def load_dataframe(path:str):
    """import csv file as pandas dataframe

    Args:
        path (str): path to csv file

    Returns:
        dataframe: dataframe of csv file
    """
    return pd.read_csv(path)

def get_image_path(path:str):
    """get list of image path in a directory

    Args:
        path (str): path to directory storing images

    Returns:
        list: a list of image path
    """
    return [os.path.join(path, filename) for filename in os.listdir(path)]

def load_image_array(path:str):
    """import image as numpy array

    Args:
        path (str): path to image file

    Returns:
        numpy.array: array of image file in form of numpy array
    """
    return np.asarray(Image.open(path))