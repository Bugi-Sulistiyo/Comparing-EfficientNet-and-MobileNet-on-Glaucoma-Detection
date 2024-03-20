import os
import random

import pandas as pd
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

def img_rand(path:str,
            label:str):
    """showing the image using streamlit fitur

    Args:
        path (str): the path of the image
        label (str): the label of the image
    
    Returns:
        np.array: an array of the image
    """
    # getting the image paths
    img_name = [os.path.join(path,
                            label,
                            filename) for filename in os.listdir(os.path.join(path, label))]
    # getting the image in array mode
    return np.asarray(Image.open(random.choice(img_name)))

def load_df(path:str):
    """getting the dataframe from the given path

    Args:
        path (str): a path of the file

    Returns:
        pd.DataFrame: a dataframe from the given path
    """
    if path.endswith('.xlsx'): # opening excel file
        return pd.read_excel(path)
    elif path.endswith('.csv'):
        return pd.read_csv(path) # opening csv file