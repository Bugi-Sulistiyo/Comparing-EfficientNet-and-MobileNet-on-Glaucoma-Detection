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

def get_file_names(path:str):
    """generate a list of file name from the given directory

    Args:
        path (str): source directory

    Returns:
        list: list of file name in the given directory
    """
    return [file_name for file_name in os.listdir(path) if os.path.isfile(os.path.join(path, file_name))]

def copy_files(source_path:str,
               destination_path:str,
               file_names:list):
    """copy files from source directory to destination directory

    Args:
        source_path (str): source directory
        destination_path (str): destination directory
        file_names (list): list of file names to be copied

    Returns:
        result_status (dict): result status of the action
    """
    result_status = {
        "Success": [],
        "Already Exists": [],
    }
    for file_name in file_names:
        try:
            shutil.copyfile(os.path.join(source_path, file_name),
                            os.path.join(destination_path, file_name))
            result_status["Success"].append(os.path.join(destination_path, file_name))
        except FileExistsError:
            result_status["Already Exists"].append(os.path.join(destination_path, file_name))
    return result_status