import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_dataframe(path:str):
    """import csv file as pandas dataframe

    Args:
        path (str): path to csv file

    Returns:
        pandas.dataframe: dataframe of csv file
    """
    if path.endswith(".xlsx"): # excel file
        return pd.read_excel(path)
    elif path.endswith(".csv"): # csv file
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

def line_chart(df:pd.DataFrame,
                x:str,
                y:list,
                title:str,
                x_axis:str,
                y_axis:str):
    """create line chart

    Args:
        df (pd.DataFrame): dataframe
        x (str): column name for x axis
        y (list): a list of column name for y axis
        title (str): title of the chart
        x_axis (str): label for x axis
        y_axis (str): label for y axis
    
    Returns:
        matplotlib.figure.Figure: line chart
    """
    fig, ax = plt.subplots()
    for col in y:
        ax.plot(df[x], round(df[col], 2), label=col)
    if len(y) > 1:
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    return fig