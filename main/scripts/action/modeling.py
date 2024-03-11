import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, EfficientNetV2M, EfficientNetV2L

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.metrics import AUC, TruePositives, TrueNegatives, FalsePositives, FalseNegatives

def get_path(scenario_names:list,
            dataset_names:list,
            fold_names:list,
            path_source:str):
    """get the path of the dataset source

    Args:
        scenario_names (list): a list of scenario names
        dataset_names (list): a list of dataset names
        fold_names (list): a list of fold names
        path_source (str): the path of the dataset source

    Returns:
        dict: a dictionary of dataset source paths
    """
    path_dataset_src = {}
    for scenario in scenario_names:
        for dataset in dataset_names:
            for fold in fold_names:
                if scenario == 'scenario_1':
                    train = 'train'
                else:
                    train = 'train_merged'

                for data_type in [train, 'val', 'test']:
                    path_dataset_src[f'{scenario}_'
                                + f'{dataset}_'
                                + f'{fold}_'
                                + f'{data_type}'] = os.path.join(path_source,
                                                                scenario,
                                                                dataset,
                                                                fold,
                                                                data_type)
    return path_dataset_src

def datagen(scenario_names:list,
            dataset_names:list,
            fold_names:list,
            path_dataset_src:dict,
            usage:str):
    """create image data generator for each dataset and fold

    Args:
        scenario_names (list): a list of scenario names
        dataset_names (list): a list of dataset names
        fold_names (list): a list of fold names
        path_dataset_src (dict): a dictionary of dataset source paths
        usage(str): the purpose of datagen with value of (training or testing)

    Returns:
        dictionary: a dictionary of image data generators
    """
    # creating the image data generator
    datagen = ImageDataGenerator(rescale=1./255)

    img_gen = {}
    image_size = (300, 300)
    for scenario in scenario_names:
        if scenario == 'scenario_1':
            train = 'train'
        else:
            train = 'train_merged'
        
        if usage == 'training':
            data_types = [train, 'val']
        elif usage == 'testing':
            data_types = ['test']
        
        for dataset in dataset_names:
            for fold in fold_names:
                for data_type in data_types:
                    print(f'Creating image data generator for {scenario} {dataset} {fold} {data_type}')
                    # create the image data generator
                    img_gen[f'{scenario}_'
                            + f'{dataset}_'
                            + f'{fold}_'
                            + f'{data_type}'] = datagen.flow_from_directory(
                                                path_dataset_src[f'{scenario}_'
                                                                + f'{dataset}_'
                                                                + f'{fold}_'
                                                                + f'{data_type}'],
                                                target_size=image_size,
                                                class_mode='binary',
                                                shuffle=True,
                                                seed=1915026018)
    
    return img_gen
    
def model_base(model_name:str,
                path_model_src:str):
    """creating the model based on the trained model name

    Args:
        model_name (str): the name of the trained model to be used. (e.g. mobilenet_v2, mobilenet_v3small, mobilenet_v3large, efficientnet_v2s, efficientnet_v2m, efficientnet_v2l)
        path_model_src (str): the path of the trained model source

    Raises:
        ValueError: if the model name is not valid or not supported

    Returns:
        tf.keras.models: a model based on the trained model name with the custom output layer
    """

    img_shape = (300,300, 3)
    # defining the trained model
    if model_name == "mobilenet_v2":
        base_model = MobileNetV2(input_shape=img_shape,
                                include_top=False,
                                weights=None
                                )
        base_model.load_weights(os.path.join(path_model_src,
                                            'mobilenet_v2.h5'))
    elif model_name == "mobilenet_v3small":
        base_model = MobileNetV3Small(input_shape=img_shape,
                                include_top=False,
                                weights=None)
        base_model.load_weights(os.path.join(path_model_src,
                                            'mobilenet_v3_small.h5'))
    elif model_name == "mobilenet_v3large":
        base_model = MobileNetV3Large(input_shape=img_shape,
                                include_top=False,
                                weights=None)
        base_model.load_weights(os.path.join(path_model_src,
                                            'mobilenet_v3_large.h5'))
    elif model_name == "efficientnet_v2s":
        base_model = EfficientNetV2S(input_shape=img_shape,
                                include_top=False,
                                weights=None)
        base_model.load_weights(os.path.join(path_model_src,
                                            'efficientnet_v2_s.h5'))
    elif model_name == "efficientnet_v2m":
        base_model = EfficientNetV2M(input_shape=img_shape,
                                include_top=False,
                                weights=None)
        base_model.load_weights(os.path.join(path_model_src,
                                            'efficientnet_v2_m.h5'))
    elif model_name == "efficientnet_v2l":
        base_model = EfficientNetV2L(input_shape=img_shape,
                                include_top=False,
                                weights=None)
        base_model.load_weights(os.path.join(path_model_src,
                                            'efficientnet_v2_l.h5'))
    else:
        raise ValueError('The model name is not valid')
    
    # adding the custom output layer
    for layer in base_model.layers:
        layer.trainable = False
    
    # adding the custom output layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    prediction_model = Dense(1, activation='sigmoid')(x)

    # creating the model
    model = Model(inputs=base_model.input, outputs=prediction_model)

    return model

def train_model(pre_trained:str,
                model_src:str,
                model_dest:str,
                model_name:str,
                datagen_train,
                datagen_val,
                dataset:str,
                fold:str):
    """training the model

    Args:
        pre_trained (str): the name of the trained model to be used. (e.g. mobilenet_v2, mobilenet_v3small, mobilenet_v3large, efficientnet_v2s, efficientnet_v2m, efficientnet_v2l)
        model_src (str): the path of the pretrained model weight is stored
        model_dest (str): the path of the trained model weight is stored
        model_name (str): the name of model file will be stored
        datagen_train (_type_): a datagenerator for training data
        datagen_val (_type_): a datagenerator for validation data
        dataset (str): the name of the dataset
        fold (str): the name of the fold

    Returns:
        _type_: the result of the training
    """
    # declare the needed variables
    ## variable for counting time
    start_time = time.perf_counter()
    ## variable for the model configuration
    optimizer = 'adam'
    loss_funct = 'binary_crossentropy'
    metrices = [AUC(name='auc'),
                TruePositives(name='tp'),
                TrueNegatives(name='tn'),
                FalsePositives(name='fp'),
                FalseNegatives(name='fn')]
    # variable for training
    epoch = 15

    # create the model structure
    tl_mnet_v2 = model_base(model_name=pre_trained,
                            path_model_src=model_src)
    # configure the model
    tl_mnet_v2.compile(optimizer=optimizer,
                    loss=loss_funct,
                    metrics=metrices)
    
    # train the model
    result = tl_mnet_v2.fit(datagen_train,
                validation_data=datagen_val,
                epochs=epoch,
                verbose=0)
    
    # save the model
    tl_mnet_v2.save(os.path.join(model_dest,
                                f'{model_name}.h5'))
    
    print(f'{dataset} {fold} finished in {round(time.perf_counter() - start_time, 2)} seconds')
    return result

def train_result(result,
                path_store:str,
                type_name:str):
    """visualize and store the training result

    Args:
        result (model.history): a model history
        path_store (str): the path to store the result
        type_name (str): the name of the model
    """
    # store the result in csv file
    result_df = pd.DataFrame(result.history)
    result_df.to_csv(os.path.join(path_store,
                                f'{type_name}.csv'),
                    index=False)

def eval_result_save(path_dest:str,
                    result:dict,
                    scenario:str,
                    model:str):
    """save the evaluation result as csv file

    Args:
        path_dest (str): the path to store the result
        result (dict): a dictionary that stored the result of evaluation
        scenario (str): the scenario name
        model (str) : the model name
    """
    pd.DataFrame(result).to_csv(os.path.join(path_dest,
                                f'{scenario}_{model}_evaluation_result.csv'),
                                index=False)

def testing_model(path_src:str,
                scenario:str,
                models:list,
                folds:list,
                datasets:list,
                datagen,
                path_dest:str):
    """evaluate the model

    Args:
        path_src (str): the path where the trained model is stored
        scenario (str): the scenario name
        models (list): a list of model name
        folds (list): a list of fold
        datasets (list): a list of used dataset to train
        datagen (_type_): the datagenrator for testing
        path_dest (str): the path that will store the result

    Returns:
        dict: a dictionary that stored the result of evaluation
    """
    for model_name in models:
        for dataset in datasets:
            # count the process time
            start_time = time.perf_counter()
            # define a variable to store the result of evalution
            result = {'model': [],
                    'scenario': [],
                    'dataset': [],
                    'fold': [],
                    'loss': [],
                    'auc': [],
                    'true_positive': [],
                    'true_negative': [],
                    'false_positive': [],
                    'false_negative': []}

            for fold in folds:
                # load the testing data
                data_test = datagen[f'{scenario}_'
                                    + f'{dataset}_'
                                    + f'{fold}_'
                                    + 'test']
                
                # load the model
                model = load_model(os.path.join(path_src,
                                                (f's{scenario.split("_")[-1]}_'
                                                + f'{model_name}_'
                                                + f'{dataset}_'
                                                + f'f{fold.split("_")[-1]}'
                                                + '.h5')))
                # evaluate the model
                loss, auc, tp, tn, fp, fn = model.evaluate(data_test,
                                                    verbose=0)

                # store the result
                result['model'].append(model_name)
                result['scenario'].append(scenario)
                result['dataset'].append(dataset)
                result['fold'].append(fold)
                result['loss'].append(loss)
                result['auc'].append(auc)
                result['true_positive'].append(tp)
                result['true_negative'].append(tn)
                result['false_positive'].append(fp)
                result['false_negative'].append(fn)

            print(f'Completed {model_name} {dataset} in {round(time.perf_counter() - start_time, 2)} seconds')
            # save the result
            pd.DataFrame(result).to_csv(os.path.join(path_dest,
                                    f'{scenario}_{model_name}_{dataset}_evaluation_result.csv'),
                                    index=False)
    return result

def get_result_data(path:str):
    """getting the result data from testing the model

    Args:
        path (str): the path of the result data is stored

    Returns:
        list: a list of result data path
    """
    path_files = {}
    try:
        files_name =  os.listdir(path)
        for file in files_name:
            if file.endswith('.csv'):
                path_files[file.split('.')[0]] = os.path.join(path,
                                                            file)
        return path_files
    except:
        return 'No data found'

def merge_result_data(path_src:str):
    """merge all the result data from testing the model

    Args:
        path_src (str): the path of the result data is stored

    Returns:
        pd.DataFrame: a dataframe that stored the result of evaluation
    """
    df_result = pd.DataFrame({
        'model': [],
        'scenario': [],
        'dataset': [],
        'fold': [],
        'loss': [],
        'auc': [],
        'true_positive': [],
        'true_negative': [],
        'false_positive': [],
        'false_negative': []
    })
    for value in get_result_data(path_src).values():
        df_result = pd.concat([df_result,
                            pd.read_csv(value)])
    return df_result