# Glaucoma Detection
This project is my research to complete my bachelor degree at Mulawarman University. The title of my research is **A Comparison of MobileNet and EfficientNet Architechture on Glaucoma Classification Based on Fundus Image**. The result  could be seen on dashboard at <a href="https://bit.ly/dashboard_skripsi_bugi">bit.ly/dashboard_skripsi_bugi</a>.

## Project Steps
### 1. Data Collecting
The dataset used for building the model is fundus image. There are 4 fundus image dataset used. This 4 datasets choosen because it is clinically diagnosed by an ophthalmologist.
- **G1020** - <a href="https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets?resource=download-directory&select=G1020">dataset</a> - <a href="https://arxiv.org/abs/2006.09158">paper</a><br>
    This dataset is collected at a private clinical practice in Kaiserslautern, Germany between year 2005 and 2017. It consisted of 1020 images from 432 patients. The images are labeled as glaucoma (296 images) and healty (724 images).

- **RIM-ONE** - <a href="https://github.com/miag-ull/rim-one-dl">dataset</a> - <a href="https://www.researchgate.net/publication/345850772_RIM-ONE_DL_A_Unified_Retinal_Image_Database_for_Assessing_Glaucoma_Using_Deep_Learning">paper</a><br>
    This dataset were taken at Universitario de Canarian Hospital and Universitario Miguel Hospital in Madrid, Spanish. It was taken in year 2011(v1), 2014(v2), 2015(v3). The one used for this project is RIM-ONE DL that make specificcally for computeri vision dataset. It consist of 485  fundus images labeled as glaucoma (172) and normal (313).

- **REFUGE** - <a href="https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets?resource=download-directory&select=REFUGE">dataset</a> - <a href="https://ieee-dataport.org/documents/refuge-retinal-fundus-glaucoma-challenge">paper</a><br>
    This dataset is a competition dataset held by MICCAI on 2018. It consist of 1200 fundus images labeled as glaucoma (1080) and normal (120).

- **PAPILA** - <a href="https://figshare.com/articles/dataset/PAPILA/14798004/1">dataset</a> - <a href="https://www.nature.com/articles/s41597-022-01388-1">paper</a><br>
    This dataset were taken at General Universitario Reino Sofia Hospital in Murcia, Spain between 2018-2020. It consist of 488 fundus images labeled as glaucoma, suspicious, and normal.

### 2. Preparation and EDA
Before all the dataset being augmented in preprocessing step, it need to be restructurize. The dataset should be restructurize because some of it have a different structure file and some of it does not devided by folder for each label. In order to do the restructurization of the file structure, the dataset must be explored and the metadata is understood. Also, data splitting process is being done in this step.

### 3. Preprocessing
The preprocessing methoed used is augmentation. The augmentation used are horizontal & vertical flipping, brightness, and CLAHE.

### 4. Training Model
There are 6 model trained using 4 dataset. I used 3 newest model from EfficientNet and MobileNet architecture each. all model is trained on every dataset and scenario defined.

### 5. Testing Model
After training model is completed, the model is immidiatly saved in .h5 format. After that, the testing script load each model and test it on every scenario.

### 6. Dashboard Development
To make the result easier to compared, I make a dashboard to show the final result.

## File Structure
- `data\` --> (hidden) store train result, model, and all important information
- `dataset\` --> (hidden) hide because have a large size. store source dataset, prepared dataset, and augmented dataset
- `lampiran\` --> store all file from data\ that will be attached on skripsi
- `main\scripts\` --> store all the script used in this project

## Replication Procedures
1. Download all the dataset mentioned above and stored it in `dataset` directory.
2. Prepare global variable stored in `.env` file. All varieble used could be seen on <a href="https://github.com/Bugi-Sulistiyo/Comparing-EfficientNet-and-MobileNet-on-Glaucoma-Detection/blob/main/var_guide.txt">`var_guide.txt`</a>
3. Run the existing script based on the numbering of existing directories and files