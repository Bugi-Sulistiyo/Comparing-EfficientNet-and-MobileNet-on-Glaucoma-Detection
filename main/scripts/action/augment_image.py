# the general image processing for training data
from tensorflow.keras.layers import RandomBrightness
from tensorflow.image import flip_left_right
from tensorflow.image import flip_up_down

# clahe image processing for training data
from tf_clahe import clahe

# visualization of the image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# other libraries
import os
import glob
import time

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

def augment_img(image:np.ndarray,
                aug_type:str):
    """augment the image based on the given type

    Args:
        image (np.ndarray): an image in the form of numpy array
        aug_type (str): the type of augmentation (h_flip, v_flip, bright)

    Returns:
        np.ndarray: the augmented image
    """
    # do the augmentation based on the type
    if aug_type == 'h_flip': # horizontal flip
        aug_img = flip_left_right(image)
    elif aug_type == 'v_flip': # vertical flip
        aug_img = flip_up_down(image)
    elif aug_type == 'bright': # brightness
        aug_brightness = RandomBrightness(factor=(0, 0.15),
                                        value_range=[.0, 1.],
                                        seed=1915026018)
        aug_img = aug_brightness(image)
    else: # no augmentation
        aug_img = tf.convert_to_tensor(image)
    
    return aug_img

def clahe_augmentation(image):
    """the augmentation for the image using clahe

    Args:
        batch_dataset (tf.data.Dataset): the batch dataset of the image

    Returns:
        tf.data.Dataset: the batch dataset of the image that has been augmented
    """
    return clahe(image,
                clip_limit=1.5)

def visualize_img(datasets:list,
                    labels:list,
                    datagen:dict,
                    aug:str,
                    clahe:bool,
                    col_mode:str,
                    scenario:str):
    """visualize the image and the augmented image

    Args:
        datasets (list): a list of the dataset names
        labels (list): a list of the label names
        datagen (dict): a dictionary of the image generator
        aug (str): the type of augmentation
        clahe (bool): the status of the clahe
        col_mode (str): the color mode of the image (grayscale or rgb)
        scenario (str): the scenario name
    """
    for dataset in datasets:
        # describe the image size
        plt.figure(figsize=(10, 10))
        for img_place, label in enumerate(labels):
            for batch_datagen in datagen[f'{dataset}_{label}']:
                # stating the augmented image type
                if clahe: # clahe augmentation
                    aug_img = clahe_augmentation(batch_datagen[0][0]).numpy()
                else: # other type augmentation
                    aug_img = augment_img(image=batch_datagen[0][0],
                                        aug_type=aug).numpy()
                
                if col_mode == 'grayscale': # for the grayscale image
                    if img_place == 0: # for image with label "normal"
                        # visualize the original image
                        plt.subplot(2, 2, img_place+1)
                        plt.title(label=label.title(),
                                    fontdict={'fontsize': 14})
                        plt.imshow(batch_datagen[0][0], cmap='gray')
                        # visualize the augmented image
                        plt.subplot(2, 2, img_place+2)
                        plt.title(label=f'Augment {label.title()}',
                                    fontdict={'fontsize': 14})
                        plt.imshow(aug_img, cmap='gray')

                    elif img_place == 1: # for image with label "glaukoma"
                        # visualize the original image
                        plt.subplot(2, 2, img_place+2)
                        plt.title(label=label.title(),
                                    fontdict={'fontsize': 14})
                        plt.imshow(batch_datagen[0][0], cmap='gray')
                        # visualize the augmented image
                        plt.subplot(2, 2, img_place+3)
                        plt.title(label=f'Augment {label.title()}',
                                    fontdict={'fontsize': 14})
                        plt.imshow(aug_img, cmap='gray')

                elif col_mode == 'rgb': # for the rgb image
                    if img_place == 0: # for image with label "normal"
                        # visualize the original image
                        plt.subplot(2, 2, img_place+1)
                        plt.title(label=label.title(),
                                    fontdict={'fontsize': 14})
                        plt.imshow(batch_datagen[0][0])
                        # visualize the augmented image
                        plt.subplot(2, 2, img_place+2)
                        plt.title(label=f'Augment {label.title()}',
                                    fontdict={'fontsize': 14})
                        plt.imshow(aug_img)
                        
                    elif img_place == 1: # for image with label "glaukoma"
                        # visualize the original image
                        plt.subplot(2, 2, img_place+2)
                        plt.title(label=label.title(),
                                    fontdict={'fontsize': 14})
                        plt.imshow(batch_datagen[0][0])
                        # visualize the augmented image
                        plt.subplot(2, 2, img_place+3)
                        plt.title(label=f'Augment {label.title()}',
                                    fontdict={'fontsize': 14})
                        plt.imshow(aug_img)
                break
        # show the visualization for each dataset
        plt.suptitle(t=f'{dataset.title()} {aug.title()} Augmentation\n{scenario.replace("_", " ").title()}',
                    y=.93,
                    verticalalignment='center',
                    fontsize=16,
                    fontweight='medium')
        plt.show()

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
                    # count the time to generate the augmented image
                    start_time = time.perf_counter()
                    print(f'Generating augmented image for {dataset}/{fold}/{d_type}...')

                    for label in labels_names:
                        img_count = len(batch_datasets[f'{dataset}_{fold}_{d_type}_{label}'])
                        # generated process per batch
                        for batch_datagen in batch_datasets[f'{dataset}_{fold}_{d_type}_{label}']:
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
    # define the dictionary to store the removed and augmented file
    rm_file = {}
    aug_file = {}
    # get the file name
    for code_name in files_code:
        # get the file name based on the code name for the removed file
        rm_file[code_name] = glob.glob(os.path.join(path_dest[scenario
                                                            + code_name[2:]],
                                                    f'[!s{scenario[-1]}_]*'))
        # get the file name based on the code name for the augmented file
        aug_file[code_name] = glob.glob(os.path.join(path_dest[scenario
                                                            + code_name[2:]],
                                                    f's{scenario[-1]}_*'))
    return rm_file, aug_file

def remove_file(files_path:list):
    """remove the files based on the given path

    Args:
        files_path (list): a list of the path of the files to be removed

    Returns:
        dict: a dictionary of the result status
    """
    # define the dictionary to store the result status
    result_status = {
        "Success": [],
        "Not Found": []
    }
    # remove the files
    for file_path in files_path:
        try:
            os.remove(file_path)
            result_status["Success"].append(file_path)
        except FileNotFoundError:
            result_status["Not Found"].append(file_path)
    return result_status