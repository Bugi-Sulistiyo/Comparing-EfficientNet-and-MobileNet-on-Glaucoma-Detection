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

def img_rand_scen(path:str,
            img_data:pd.DataFrame,
            scenario:int,
            augment_type:str):
    """showing the image using streamlit fitur for specific scenario and augmentation type

    Args:
        path (str): the path of the image
        img_data (pd.DataFrame): the dataframe of the image data
        scenario (int): the scenario of the image
        augment_type (str): the augmentation type of the image

    Returns:
        np.array: an array of the image
    """
    # getting the files name for specific scenario and augmentation type
    img_name = list(img_data.loc[(img_data.scenario == scenario)
                                    & (img_data.augmentation == augment_type),
                                'file_name'].values)
    # getting the image paths
    img_name = [os.path.join(path, img) for img in img_name]
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

def img_scen_list(df_path:str,
                img_path:str,
                scenario:int,
                augment_type:str):
    """getting the list of image files for specific scenario and augmentation type

    Args:
        df_path (str): a path of the csv file
        img_path (str): a path of the image
        scenario (int): the selected scenario
        augment_type (str): the selected augmentation type

    Returns:
        np.array: an array of the image
    """
    # get the dataframe that contains the image data
    df_img = load_df(df_path)
    # get the files name for specific scenario and augmentation type
    img_name =  list(df_img.loc[(df_img.scenario == scenario)
                            & (df_img.augmentation == augment_type),
                            'file_name'].values)
    # get the image paths
    img_name = [os.path.join(img_path, img) for img in img_name]
    # get the image in array mode
    return np.asarray(Image.open(random.choice(img_name)))

def string_upper(val):
    """changing the string to uppercase

    Args:
        val (str/float/int): the value that will be changed

    Returns:
        str/int: the value that has been changed
    """
    if type(val) == str:
        if val == 'auc':
            return val.upper()
        else:
            return val.replace('_', ' ').title()
    else:
        return round(val, 4)

def content(path:str):
    """getting the content for filling the dashboard

    Args:
        path (str): the path of the txt file

    Returns:
        str: the content of the txt file
    """
    file = open(f'{path}.txt', 'r')
    return file.read()