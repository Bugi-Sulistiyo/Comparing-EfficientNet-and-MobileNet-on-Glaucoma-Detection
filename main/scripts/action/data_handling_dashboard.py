import os
import pandas as pd
import numpy as np
from PIL import Image

def load_dataframe(path):
    return pd.read_csv(path)

def get_image_path(path):
    return [os.path.join(path, filename) for filename in os.listdir(path)]

def load_image_array(path):
    return np.asarray(Image.open(path))