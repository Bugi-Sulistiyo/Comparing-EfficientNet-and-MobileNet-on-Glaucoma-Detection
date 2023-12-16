# handling image file
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.utils import save_img

# basic image processing for training, validation, and testing
from tensorflow.keras.layers import Resizing
from tensorflow.keras.layers import Rescaling

# the general image processing for training data
from tensorflow.image import stateless_random_flip_left_right
from tensorflow.image import stateless_random_flip_up_down
from tensorflow.keras.layers import RandomBrightness
from tensorflow.keras.layers import RandomRotation

# the grey scale image processing for training data
from tensorflow.image import rgb_to_grayscale

# clahe image processing for training data
from tf_clahe import clahe

# visualization of the image
import matplotlib.pyplot as plt