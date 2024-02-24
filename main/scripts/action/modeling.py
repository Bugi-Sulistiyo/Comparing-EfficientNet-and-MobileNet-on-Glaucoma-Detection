import tensorflow as tf
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, EfficientNetV2M, EfficientNetV2L

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.models import Model

def datagen(scenario_names:list,
            dataset_names:list,
            fold_names:list,
            path_dataset_src:dict,
            image_size:dict,
            col_mode:dict):
    """create image data generator for each dataset and fold

    Args:
        scenario_names (list): a list of scenario names
        dataset_names (list): a list of dataset names
        fold_names (list): a list of fold names
        path_dataset_src (dict): a dictionary of dataset source paths
        image_size (dict): a dictionary of image sizes
        col_mode (dict): a dictionary of color modes

    Returns:
        dictionary: a dictionary of image data generators
    """
    # creating the image data generator
    datagen = ImageDataGenerator(rescale=1./255)

    img_gen = {}
    for scenario in scenario_names:
        for dataset in dataset_names:
            for fold in fold_names:

                if scenario == 'scenario_1':
                    train = 'train'
                else:
                    train = 'train_augmented'

                for data_type in [train, 'val', 'test']:
                    # create the image data generator
                    img_gen[f'{scenario}_'
                            + f'{dataset}_'
                            + f'{fold}_'
                            + f'{data_type}'] = datagen.flow_from_directory(
                                                path_dataset_src[f'{scenario}_'
                                                                + f'{dataset}_'
                                                                + f'{fold}_'
                                                                + f'{data_type}'],
                                                target_size=image_size[dataset],
                                                color_mode=col_mode[scenario],
                                                class_mode='binary',
                                                shuffle=True,
                                                seed=1915026018)
    
    return img_gen
    
def model_base(model_name:str,
                img_shape:tuple,
                path_model_src:str):
    """creating the model based on the trained model name

    Args:
        model_name (str): the name of the trained model to be used. (e.g. mobilenet_v2, mobilenet_v3small, mobilenet_v3large, efficientnet_v2s, efficientnet_v2m, efficientnet_v2l)
        img_shape (tuple): the shape of the input image with the format (height, width, channels)
        path_model_src (str): the path of the trained model source

    Raises:
        ValueError: if the model name is not valid or not supported

    Returns:
        tf.keras.models: a model based on the trained model name with the custom output layer
    """
    # defining the trained model
    if model_name == "mobilenet_v2":
        base_model = MobileNetV2(input_shape=img_shape,
                                include_top=False,
                                weights=None)
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