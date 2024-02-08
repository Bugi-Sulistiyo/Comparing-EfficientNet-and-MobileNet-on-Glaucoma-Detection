# handling image file
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.utils import save_img

# basic image processing for training, validation, and testing
from tensorflow.keras.layers import Resizing
from tensorflow.keras.layers import Rescaling

# the general image processing for training data
from tensorflow.keras.layers import RandomBrightness
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.layers import RandomFlip

# clahe image processing for training data
from tf_clahe import clahe

# visualization of the image
import matplotlib.pyplot as plt

# other libraries
import os
import numpy as np
import tensorflow as tf

def get_image(source_path:str,
            img_width:int,
            img_height:int,
            color_mode:str,
            batch_size:int):
    """import images from the given directory.

    Args:
        source_path (str): the path where the image files are located
        img_width (int): the width of the image
        img_height (int): the height of the image
        color_mode (str): the color mode of the image (grayscale or rgb)
        batch_size (int): the size of the batch

    Returns:
        tf.data.Dataset: the image in the form of tensorflow dataset
    """
    return image_dataset_from_directory(directory = source_path,
                                        image_size = (img_height, img_width),
                                        color_mode = color_mode,
                                        batch_size = batch_size,
                                        shuffle = True,
                                        seed = 1915026018,
                                        labels='inferred',
                                        label_mode = 'binary')

def visualize_image(batch_dataset:tf.data.Dataset,
                    figure_column:int,
                    figure_row:int,
                    figure_size_height:int,
                    figure_size_width:int,
                    label_names:list):
    """visualize the image from the batch dataset in the form of grid

    Args:
        batch_dataset (tf.data.Dataset): the batch dataset of the image
        figure_column (int): the number of column in the grid
        figure_row (int): the number of row in the grid
        figure_size_height (int): the height of the figure
        figure_size_width (int): the width of the figure
    """
    plt.figure(figsize=(figure_size_width, figure_size_height))
    for images, labels in batch_dataset.take(1):
        for i in range(figure_column * figure_row):
            plt.subplot(figure_row, figure_column, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(label_names[int(labels[i])])
            plt.axis("off")
    plt.show()

def basic_augment(batch_dataset:tf.data.Dataset,
                flip_mode:str,
                clockwise_rotation:float,
                counter_clockwise_rotation:float,
                bright_lower_bound:float,
                bright_upper_bound:float,
                seed:int = 1915026018):
    """the basic augmentation for the image. the augmentation includes flipping, rotation, and brightness

    Args:
        batch_dataset (tf.data.Dataset): the batch dataset of the image
        flip_mode (str): the mode of flipping (horizontal, vertical, or horizontal_and_vertical)
        clockwise_rotation (float): the factor of clockwise rotation. the value is in the range of (.0 - -1.)
        counter_clockwise_rotation (float): the factor of counter clockwise rotation. the value is in the range of (.0 - 1.)
        bright_lower_bound (float): the lower bound of brightness. the value is in the range of (.0 - -1.)
        bright_upper_bound (float): the upper bound of brightness. the value is in the range of (.0 - 1.)
        seed (int, optional): the seed for random state. Defaults to 1915026018.

    Returns:
        tf.data.Dataset: the batch dataset of the image that has been augmented
    """
    image, label = batch_dataset
    basic_augment_layer = tf.keras.Sequential([
        Rescaling(1./255),
        RandomFlip(mode=flip_mode,
                    seed=seed),
        RandomRotation(factor=(clockwise_rotation,
                                counter_clockwise_rotation),
                        seed=seed),
        RandomBrightness(factor=(bright_lower_bound,
                                bright_upper_bound),
                        value_range=[0,1],
                        seed=seed)
    ])
    return basic_augment_layer(image), label

def clahe_augmentation(image,
                        clip_limit):
    """the augmentation for the image using clahe

    Args:
        batch_dataset (tf.data.Dataset): the batch dataset of the image

    Returns:
        tf.data.Dataset: the batch dataset of the image that has been augmented
    """
    # @tf.function(experimental_compile=True)
    # def fast_clahe(image):
    #     return clahe(image, gpu_optimized=True)
    # return fast_clahe(image)
    return clahe(image,
                clip_limit=clip_limit)

# def save_image(image:np.array,
#                 destination_path:str,
#                 data_format:str='channels_last',
#                 file_format:str='jpg'):
    