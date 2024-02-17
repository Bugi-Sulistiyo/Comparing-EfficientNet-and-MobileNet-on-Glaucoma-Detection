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

from tensorflow.image import flip_left_right
from tensorflow.image import flip_up_down

# clahe image processing for training data
from tf_clahe import clahe

# visualization of the image
import matplotlib.pyplot as plt

# other libraries
import os
import glob
import time
import numpy as np
import tensorflow as tf

def create_directory(path_dict:dict):
    """create directory based on the given dictionary

    Args:
        path_dict (dict): a dictionary of the path with the key as the name of the directory and the value as the path

    Returns:
        dict: a dictionary of the result status
    """
    result_status = {
        "Success": [],
        "Already Exists": []
    }
    # create the directory
    for key, value in path_dict.items():
        try:
            os.makedirs(value)
            result_status["Success"].append(key)
        except FileExistsError:
            result_status["Already Exists"].append(key)
    
    return result_status

def show_augmented_img(image:np.ndarray,
                        augment_type:str,
                        color_mode:str='rgb'):
    """visualize the augmented image with comparison to the original image

    Args:
        image (np.ndarray): an image in the form of numpy array
        augment_type (str): the type of augmentation (h_flip, v_flip, bright, rot)
        color_mode (str, optional): the color mode of the image (grayscale or rgb). Defaults to 'rgb'.
    """
    # define the figure size
    plt.figure(figsize=(10, 10))
    # do the augmentation based on the type
    if augment_type == 'h_flip': # horizontal flip
        aug_img = flip_left_right(image)
    elif augment_type == 'v_flip': # vertical flip
        aug_img = flip_up_down(image)
    elif augment_type == 'bright': # brightness
        aug_rotate = RandomBrightness(factor=(0, 0.15),
                                        value_range=[.0, 1.],
                                        seed=1915026018)
        aug_img = aug_rotate(image)
    elif augment_type == 'rot': # rotation
        aug_rotate = RandomRotation(factor=(-0.5, 0.5), seed=1915026018)
        aug_img = aug_rotate(image)
    else: # no augmentation
        aug_img = tf.convert_to_tensor(image)

    # visualize the image
    if color_mode == 'grayscale':
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(aug_img.numpy(), cmap='gray')
    elif color_mode == 'rgb':
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(aug_img.numpy())
    
    plt.show()

def clahe_augmentation(image):
    """the augmentation for the image using clahe

    Args:
        batch_dataset (tf.data.Dataset): the batch dataset of the image

    Returns:
        tf.data.Dataset: the batch dataset of the image that has been augmented
    """
    ## for the fast version of the clahe
    # @tf.function(experimental_compile=True)
    # def fast_clahe(image):
    #     return clahe(image, gpu_optimized=True)
    # return fast_clahe(image)
    ## for the normal version of the clahe
    return clahe(image,
                clip_limit=1.5)

def generate_aug_img(dataset_names:list,
                    fold_names:list,
                    labels_names:list,
                    batch_datasets:dict,
                    data_type:str):
    """generate the augmented image from the image data generator

    Args:
        dataset_names (list): a list of the dataset names
        fold_names (list): a list of the fold names
        labels_names (list): a list of the label names
        batch_datasets (dict): a dictionary of the batch datasets image generator
        data_type (str): the type of data would be generated (train or val_test)
    """
    # generate the augmented image for the training data
    if data_type == 'train':
        # create the variabel to avoid infinite loop
        exit_count = 0
        for dataset in dataset_names:
            for fold in fold_names:
                for label in labels_names:
                    # count the time to generate the augmented image
                    start_time = time.perf_counter()
                    print(f'Generating augmented image for {dataset}/{fold}/{label}...')
                    img_count = len(batch_datasets[f'{dataset}_{fold}_{label}'])

                    # generated process per batch
                    for batch_datagen in batch_datasets[f'{dataset}_{fold}_{label}']:
                        exit_count += 1
                        if exit_count == img_count:
                            exit_count = 0
                            break
                    
                    print(f'Elapsed time: {time.perf_counter() - start_time:.2f} seconds')
    # generate the augmented image for the validation and testing data
    elif data_type == 'val_test':
        # create the variabel to avoid infinite loop
        exit_count = 0
        for dataset in dataset_names:
            for fold in fold_names:
                for d_type in ['val', 'test']:
                    for label in labels_names:
                        # count the time to generate the augmented image
                        start_time = time.perf_counter()
                        print(f'Generating augmented image for {dataset}/{fold}/{label}...')
                        img_count = len(batch_datasets[f'{dataset}_{fold}_{d_type}_{label}'])

                        # generated process per batch
                        for batch_datagen in batch_datasets[f'{dataset}_{fold}_{label}']:
                            exit_count += 1
                            if exit_count == img_count:
                                exit_count = 0
                                break
                        
                        print(f'Elapsed time: {time.perf_counter() - start_time:.2f} seconds')

def get_file(files_code:list,
            path_dest:dict,
            scenario:str):
    """get image name that has been augmented and removed for val and test data

    Args:
        files_code (list): a list of code name of the files
        path_dest (dict): a dictionary that stores the path of the augmented files
        scenario (str): the scenario name

    Returns:
        dict: a dictionary of the removed file and augmented file
    """
    rm_file = {}
    aug_file = {}
    for code_name in files_code:
        rm_file[code_name] = glob.glob(os.path.join(path_dest[scenario
                                                            + code_name[2:]],
                                                    f'[!s3_]*'))
        aug_file[code_name] = glob.glob(os.path.join(path_dest[scenario
                                                            + code_name[2:]],
                                                    f's3_*'))
    return rm_file, aug_file

def remove_file(files_path:list):
    """remove the files based on the given path

    Args:
        files_path (list): a list of the path of the files to be removed

    Returns:
        dict: a dictionary of the result status
    """
    result_status = {
        "Success": [],
        "Not Found": []
    }
    for file_path in files_path:
        try:
            os.remove(file_path)
            result_status["Success"].append(file_path)
        except FileNotFoundError:
            result_status["Not Found"].append(file_path)
    return result_status

# =================             Experimentation             =================
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