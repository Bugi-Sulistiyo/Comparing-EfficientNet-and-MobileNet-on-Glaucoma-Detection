import os
import shutil

def create_destination_directory(path:str,
                                 dataset_names:list,
                                 label_names:list):
    """action to create destination directory for dataset

    Args:
        path (str): path where the dataset will be stored
        dataset_names (list): dataset names
        label_names (list): label names of the dataset
    
    Returns:
        result_status (dict): result status of the action
    """
    result_status = {
        "Success": [],
        "Already Exists": [],
    }
    for dataset_name in dataset_names:
        for label_name in label_names:
            try:
                os.makedirs(os.path.join(path,
                                         dataset_name,
                                         label_name))
                result_status["Success"].append(os.path.join(path,
                                                             dataset_name,
                                                             label_name))
            except FileExistsError:
                result_status["Already Exists"].append(os.path.join(path,
                                                                        dataset_name,
                                                                        label_name))
    return result_status

def get_file_name(path):
    return os.path.basename(path)