import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

import streamlit as st

# global variable
## path
PATH_DATA = r'./../../../data'
PATH_AUTHOR_DATA = os.path.join(PATH_DATA, 'author_info.csv')
PATH_DATASET_DATA = os.path.join(PATH_DATA, 'dataset_info.csv')
PATH_ARSITEKTUR_DATA = os.path.join(PATH_DATA, 'arsitektur_info.csv')
PATH_DATASET = r'./../../../dataset'
PATH_RIMONE = os.path.join(PATH_DATASET, 'RIM-ONE_DL_images/partitioned_by_hospital/training_set')
PATH_G1020 = os.path.join(PATH_DATASET, 'G1020/Images')
PATH_REFUGE = os.path.join(PATH_DATASET, 'REFUGE/Images_Square')
PATH_PAPILA = os.path.join(PATH_DATASET, 'Papila/FundusImages')
## others
LOREM_IPSUM100 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec vitae lobortis arcu. Quisque pulvinar venenatis libero, vitae semper metus euismod pulvinar. Nam justo ligula, laoreet ut malesuada et, egestas eu quam. Quisque pharetra bibendum ultricies. Etiam condimentum risus enim. Mauris semper, mauris sed aliquet blandit, purus tellus porta velit, vitae auctor magna magna sed neque. Nunc vestibulum tincidunt ligula, ut scelerisque mi molestie non. Aenean nec maximus arcu. Integer gravida tellus sit amet rhoncus consequat. Integer scelerisque placerat felis at dignissim. Sed eget auctor sem, rhoncus condimentum odio. Nullam sem dolor, mollis id ultrices sit amet, sollicitudin nec nulla. Vivamus."
THESIS_TITLE = "Perbandingan Arsitektur MobileNet dan EfficientNet pada Klasifikasi Glaukoma Berdasarkan Citra Fundus"

# data handling
def load_dataframe(path):
    return pd.read_csv(path)

def load_image_array(path):
    return np.asarray(Image.open(path))

# dasboard section
## summary and title
with st.container():
    st.title("Laporan Evaluasi Model Klasifikasi Glaukoma",
             anchor="judul-laporan")
    st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM100}</div>",
                unsafe_allow_html=True)

# info dataset
with st.container():
    dataset_info_df = load_dataframe(PATH_DATASET_DATA)
    st.header("Informasi Dataset", anchor="info-dataset")
    st.dataframe(dataset_info_df.loc[:, ~dataset_info_df.columns.isin(['Link', 'Sumber'])])


## tab - model evaluation
with st.container():
    # st.header("Hasil Evaluasi Model per Dataset", anchor="hasil-evaluasi-model")
    tab_rimone, tab_g1020, tab_refuge, tab_papila = st.tabs(["RIM-ONE",
                                                             "G1020",
                                                             "REFUGE",
                                                             "PAPILA"])
    dataset_info_df = load_dataframe(PATH_DATASET_DATA)

    ### rimone dataset
    with tab_rimone:
        st.subheader("Contoh Gambar", anchor="rimone-dataset")



## summary and conclusion
with st.container():
    st.header("Kesimpulan", anchor="kesimpulan")
    st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM100}</div>",
                unsafe_allow_html=True)

## sidebar
st.sidebar.header("Judul", anchor="Info Penelitian")
st.sidebar.markdown(f'{THESIS_TITLE}')

st.sidebar.header("Penulis")
author_data = load_dataframe(PATH_AUTHOR_DATA)
for row in load_dataframe(PATH_AUTHOR_DATA).iterrows():
    with st.sidebar.expander(row[1][-1]):
        st.markdown(f"**{row[1][0]}**")
        st.text(f"{row[1][1]}")
        st.markdown(f"{row[1][2]}")

st.sidebar.header("Penelitian")
for path in [["Dataset", PATH_DATASET_DATA],
             ["Arsitektur", PATH_ARSITEKTUR_DATA]]:
    with st.sidebar.expander(path[0]):
        for row in load_dataframe(path[1]).iterrows():
            st.markdown(f"""**[{row[1][0]}]({row[1][1]} "{row[1][2]}")**""")