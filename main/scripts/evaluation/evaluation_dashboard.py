import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.environ.get('PATH_CUSTOM_MODULES'))


import matplotlib.pyplot as plt
import streamlit as st

import data_handling_dashboard as dh

# global variable
## path
PATH_DATA = os.environ.get('PATH_DATA_DASHBOARD')
PATH_AUTHOR_DATA = os.path.join(PATH_DATA, 'author_info.csv')
PATH_DATASET_DATA = os.path.join(PATH_DATA, 'dataset_info.csv')
PATH_ARSITEKTUR_DATA = os.path.join(PATH_DATA, 'arsitektur_info.csv')
PATH_DATASET = os.environ.get('PATH_DATASET')
PATH_RIMONE = os.path.join(PATH_DATASET, 'rimone')
PATH_G1020 = os.path.join(PATH_DATASET, 'g1020')
PATH_REFUGE = os.path.join(PATH_DATASET, 'refuge')
PATH_PAPILA = os.path.join(PATH_DATASET, 'papila')
## others
LOREM_IPSUM100 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec vitae lobortis arcu. Quisque pulvinar venenatis libero, vitae semper metus euismod pulvinar. Nam justo ligula, laoreet ut malesuada et, egestas eu quam. Quisque pharetra bibendum ultricies. Etiam condimentum risus enim. Mauris semper, mauris sed aliquet blandit, purus tellus porta velit, vitae auctor magna magna sed neque. Nunc vestibulum tincidunt ligula, ut scelerisque mi molestie non. Aenean nec maximus arcu. Integer gravida tellus sit amet rhoncus consequat. Integer scelerisque placerat felis at dignissim. Sed eget auctor sem, rhoncus condimentum odio. Nullam sem dolor, mollis id ultrices sit amet, sollicitudin nec nulla. Vivamus."
THESIS_TITLE = "Perbandingan Arsitektur MobileNet dan EfficientNet pada Klasifikasi Glaukoma Berdasarkan Citra Fundus"
LABELS_USADE = ["Normal", "Glaukoma"]

# dasboard section
## summary and title
with st.container():
    st.title("Laporan Evaluasi Model Klasifikasi Glaukoma",
             anchor="judul-laporan")
    st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM100}</div>",
                unsafe_allow_html=True)

# info dataset
with st.container():
    dataset_info_df = dh.load_dataframe(PATH_DATASET_DATA)
    st.header("Informasi Dataset", anchor="info-dataset")
    st.dataframe(dataset_info_df.loc[:, ~dataset_info_df.columns.isin(['Link', 'Sumber'])])


## tab - model evaluation
with st.container():
    # st.header("Hasil Evaluasi Model per Dataset", anchor="hasil-evaluasi-model")
    tab_rimone, tab_g1020, tab_refuge, tab_papila = st.tabs(["RIM-ONE",
                                                             "G1020",
                                                             "REFUGE",
                                                             "PAPILA"])
    dataset_info_df = dh.load_dataframe(PATH_DATASET_DATA)

    ### rimone dataset
    with tab_rimone:
        st.subheader("Contoh Gambar", anchor="rimone-dataset")
        st.image(dh.get_image_path(PATH_RIMONE))


## summary and conclusion
with st.container():
    st.header("Kesimpulan", anchor="kesimpulan")
    st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM100}</div>",
                unsafe_allow_html=True)

## sidebar
st.sidebar.header("Judul", anchor="Info Penelitian")
st.sidebar.markdown(f'{THESIS_TITLE}')

st.sidebar.header("Penulis")
author_data = dh.load_dataframe(PATH_AUTHOR_DATA)
for row in dh.load_dataframe(PATH_AUTHOR_DATA).iterrows():
    with st.sidebar.expander(row[1][-1]):
        st.markdown(f"**{row[1][0]}**")
        st.text(f"{row[1][1]}")
        st.markdown(f"{row[1][2]}")

st.sidebar.header("Penelitian")
for path in [["Dataset", PATH_DATASET_DATA],
             ["Arsitektur", PATH_ARSITEKTUR_DATA]]:
    with st.sidebar.expander(path[0]):
        for row in dh.load_dataframe(path[1]).iterrows():
            st.markdown(f"""**[{row[1][0]}]({row[1][1]} "{row[1][2]}")**""")