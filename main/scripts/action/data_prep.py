import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def create_destination_directory(path:str,
                                dataset_names:list,
                                label_names:list,
                                create_type:str="merge"):
    """action to create destination directory for dataset

    Args:
        path (str): path where the dataset will be stored
        dataset_names (list): dataset names
        label_names (list): label names of the dataset
        create_type (str, optional): type of creation (merge/usage). Defaults to "merge".
    
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
                if create_type == "merge":
                    os.makedirs(os.path.join(path,
                                            dataset_name,
                                            label_name))
                    result_status["Success"].append(os.path.join(path,
                                                                dataset_name,
                                                                label_name))
                elif create_type == "usage":
                    for dataset_type in ['train', 'val', 'test']:
                        os.makedirs(os.path.join(path,
                                                dataset_type,
                                                label_name))
                        result_status["Success"].append(os.path.join(path,
                                                                    dataset_type,
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

def fname_dict_df(file_names:dict, label_names:list):
    """convert file names dictionary to dataframe

    Args:
        file_names (dict): a dictionary of file names with label names as keys and list of file names as values
        label_names (list): a list of label names. only two label names are allowed

    Returns:
        DataFrame: dataframe of file names with label names
    """
    df_file_name = pd.DataFrame(file_names[label_names[0]]
                                + file_names[label_names[1]])
    df_file_name.rename(columns={0: "file_name"}, inplace=True)
    df_file_name.loc[df_file_name.file_name.isin(file_names[label_names[0]]),
                    "label"] = label_names[0]
    df_file_name.loc[df_file_name.file_name.isin(file_names[label_names[1]]),
                    "label"] = label_names[1]
    return df_file_name

def split_file(val_size:float, test_size:float,
                random_state:int,
                df_file_name:pd.DataFrame,
                source_path:str,
                destination_path:str):
    """split file names into train, val, and test. also copy the files into the destination directory

    Args:
        val_size (float): the size of validation set in the range of 0.0 - 1.0
        test_size (float): the size of test set in the range of 0.0 - 1.0
        random_state (int): the seed for random state
        df_file_name (pd.DataFrame): a dataframe of file names with column "file_name" and "label"
        source_path (str): path to the source directory
        destination_path (str): path to the destination directory

    Returns:
        dict: a dictionary of result status
    """
    # prepare the splitting variables
    stratified_kfold_train_test = StratifiedKFold(n_splits=int(1/test_size),
                                                shuffle=True,
                                                random_state=random_state)
    stratified_kfold_val_test = StratifiedKFold(n_splits=int(1/val_size),
                                                shuffle=True,
                                                random_state=random_state)

    result = {}
    # main splitting process
    for folds, (train_val_index, test_index) in enumerate(stratified_kfold_train_test.split(df_file_name.file_name,
                                                                                            df_file_name.label)):
        # split the file names into train, val, and test
        df_train_val_name = df_file_name.iloc[train_val_index]
        train_index, val_index = next(stratified_kfold_val_test.split(df_train_val_name.file_name,
                                                                        df_train_val_name.label))
        df_train_name = df_train_val_name.iloc[train_index]
        df_val_name = df_train_val_name.iloc[val_index]
        df_test_name = df_file_name.iloc[test_index]

        # create destination directory
        dir_result = create_destination_directory(path = os.path.join(destination_path,
                                                                        f'fold_{folds+1}'),
                                                dataset_names = "empty",
                                                label_names = df_file_name.label.unique(),
                                                create_type = "usage")
        
        # copy files
        copy_result = {}
        for dataset_type, df_name in zip(["train", "val", "test"],
                                        [df_train_name, df_val_name, df_test_name]):
            copy_result[f'{dataset_type}'] = {}
            for label_name in df_file_name.label.unique():
                copy_result[f'{dataset_type}'][f'{label_name}'] = copy_files(source_path = os.path.join(source_path,
                                                                                                        label_name),
                                                                            destination_path = os.path.join(destination_path,
                                                                                                            f'fold_{folds+1}',
                                                                                                            dataset_type,
                                                                                                            label_name),
                                                                            file_names = df_name.loc[df_name.label == label_name,
                                                                                                    "file_name"].values.tolist())
        
        result[f'fold {folds+1}'] = {"create directory": dir_result,
                                    "copy files": copy_result}
        
    return result