## Main Scripts
This directory store all the necesessary scripts. All the scripts stored in a folder based on their stage in this project. Almost all scripts stored in lettered directory import function from scripts stored in `action\` directory

## File Explanation
### a. preparation
- `0. prep_data_dashboard.ipynb` --> prepare the data for dashboard prototype
- `1. merge_image_file.ipynb` --> (prepare) merge and restructure each dataset file structure into a general more understanable structure based on respective metadata
- `2. analyze_image_file.ipynb` --> analyze image dimension on each dataset
### b. preprocessing
- `1. data_split.ipynb` --> split dataset into train set, val set, and test set. also implement the cross-validation methode using 5 folds.
- `2. augmentation.ipynb` --> implement the augmentation for scenario 2 and 3 on each dataset.
- `3. visualize_aug_img.ipynb` --> save and example of each dataset and augmentation scenario as image file.
- `4. augment_adjustment.ipynb` --> merge the augmented image with the original image on training set
### c. modeling
- `1. training_model.ipynb` --> build and train each model
- `2. testing_model.ipynb` --> test each model
- `3. result_handling.ipynb` --> handle all the testing log result and save it to csv file
- `paper_needs.ipynb` --> create a visualization for paper
### d. evaluation
- `evaluation_dashboard.py` --> streamlit dashboard for showing the final result. still prototype
- `handling_data.ipynb` --> handle all the necessary data for dashboard assets
### action
- `analyze_image.py` --> store all function for analyze dataset
- `augment_image.py` --> store all function for augmentation process
- `data_prep.py` --> store all function for preparing the data (data preparation)
- `evaluation.py` --> store all function for evalution related scripts
- `modeling.py` --> store all function for training and testing related scripts
### experiment
- `augmentation.ipynb` --> experiment the visualization and augmentation method
- `test.ipynb` --> experiment all kind of things