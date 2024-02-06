# handling the metadata of the image in form of dataframe
import pandas as pd
# for visualization of the image and the metadata
import matplotlib.pyplot as plt
# for handling the image file
from PIL import Image
# other libraries
import os

def create_metadata(dataset_names:list,
                    dataset_labels:list,
                    file_names:dict,
                    path_source:str):
    """create metadata in form of dataframe from the given image files

    Args:
        dataset_names (list): a list of dataset names
        dataset_labels (list): a list of label names
        file_names (dict): a nested dictionary of file names divided by dataset and label
        path_source (str): the path where the image files are located

    Returns:
        pd.DataFrame: the final metadata in form of dataframe
    """

    metadata_images = pd.DataFrame(columns=['dataset',
                                        'label',
                                        'file_name',
                                        'width',
                                        'height',
                                        'color_space',
                                        'path'])

    for dataset_name in dataset_names:
        for label_name in dataset_labels:
            for file_name in file_names[dataset_name][label_name]:
                path_image = os.path.join(path_source,
                                        dataset_name,
                                        label_name,
                                        file_name)
                image = Image.open(path_image)
                metadata_images.loc[len(metadata_images)] = [dataset_name,
                                                            label_name,
                                                            file_name,
                                                            image.width,
                                                            image.height,
                                                            image.mode,
                                                            path_image]
    return metadata_images

def visualize_data_distribution(metadata:pd.DataFrame,
                                min_line_height:int,
                                max_line_height:int,
                                bin_count:int):
    """visualize the distribution of the image width

    Args:
        metadata (pd.DataFrame): the metadata of the image in form of dataframe
        min_line_height (int): the minimum height of the vertical line
        max_line_height (int): the maximum height of the vertical line
        bin_count (int): the number of bins in the histogram

    Returns:
        list: a list of the median, mean, min, and max of the image width distribution
    """

    plt.vlines(metadata.width.mean(),
                min_line_height, max_line_height,
                color='green', linestyles='dashed',
                label='mean')
    plt.vlines(metadata.width.median(),
                min_line_height, max_line_height,
                color='red', linestyles='dashed',
                label='median')
    plt.vlines(metadata.width.min(),
                min_line_height, max_line_height,
                color='blue', linestyles='solid',
                label='min')
    plt.vlines(metadata.width.max(),
                min_line_height, max_line_height,
                color='blue', linestyles='solid',
                label='max')
    plt.hist(metadata.width,
            bins=bin_count)
    plt.legend()
    plt.show()

    return {'median': metadata.width.median(),
            'mean': metadata.width.mean(),
            'min': metadata.width.min(),
            'max': metadata.width.max()}