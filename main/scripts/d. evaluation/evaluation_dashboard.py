import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.environ.get('PATH_CUSTOM_MODULES'))

import streamlit as st

import evaluation

# global variable
## path
### main path
PATH_DATA = os.environ.get('PATH_DATA_DASHBOARD')
PATH_DATASET = os.environ.get('PATH_DATASET_COMBINED')
### data path
PATH_AUTHOR_DATA = os.path.join(PATH_DATA, 'author_info.csv')
PATH_DATASET_DATA = os.path.join(PATH_DATA, 'dataset_info.csv')
PATH_ARSITEKTUR_DATA = os.path.join(PATH_DATA, 'arsitektur_info.csv')
### dataset path
PATH_RIMONE = os.path.join(PATH_DATASET, 'rimone')
PATH_G1020 = os.path.join(PATH_DATASET, 'g1020')
PATH_REFUGE = os.path.join(PATH_DATASET, 'refuge')
PATH_PAPILA = os.path.join(PATH_DATASET, 'papila')
## others
LOREM_IPSUM100 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec vitae lobortis arcu. Quisque pulvinar venenatis libero, vitae semper metus euismod pulvinar. Nam justo ligula, laoreet ut malesuada et, egestas eu quam. Quisque pharetra bibendum ultricies. Etiam condimentum risus enim. Mauris semper, mauris sed aliquet blandit, purus tellus porta velit, vitae auctor magna magna sed neque. Nunc vestibulum tincidunt ligula, ut scelerisque mi molestie non. Aenean nec maximus arcu. Integer gravida tellus sit amet rhoncus consequat. Integer scelerisque placerat felis at dignissim. Sed eget auctor sem, rhoncus condimentum odio. Nullam sem dolor, mollis id ultrices sit amet, sollicitudin nec nulla. Vivamus."
LOREM_IPSUM10 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec vitae lobortis arcu"
THESIS_TITLE = "Perbandingan Arsitektur MobileNet dan EfficientNet pada Klasifikasi Glaukoma Berdasarkan Citra Fundus"
LABELS_USED = ["normal", "glaukoma"]

# function section
def show_img(path:str,
            label:str):
    """showing a random image from the given path and label using streamlit fitur

    Args:
        path (str): a string of path of the image
        label (str): the label of the image
    """
    st.image(evaluation.img_rand(path=path, label=label),
            caption=label.title(),
            use_column_width=True)

def example_img(path:str,
                label:list):
    """arranging the example image in the dashboard from the given template

    Args:
        path (str): a string of path of the image
        label (list): a list of label
    """

    col_normal_image, col_glaucoma_image = st.columns(2)
    with col_normal_image:
        show_img(path=path, label=label[0])
    with col_glaucoma_image:
        show_img(path=path, label=label[1])

# dasboard section
## summary and title
with st.container():
    st.title("Evaluasi Model Klasifikasi Glaukoma",
                anchor="judul-laporan")
    st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM100}</div>",
                unsafe_allow_html=True)

## info dataset
with st.container():
    ### dataset summary
    dataset_info_df = evaluation.load_df(path=PATH_DATASET_DATA)
    st.header("Dataset", anchor="info-dataset")
    st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM10}</div><br>",
                unsafe_allow_html=True)
    st.dataframe(dataset_info_df.loc[:, ~dataset_info_df.columns.isin(['Link', 'Sumber'])])

    ### example image
    st.subheader('Contoh Data Citra', anchor=f'image-dataset')
    tab_rimone, tab_g1020, tab_refuge, tab_papila = st.tabs(["RIM-ONE",
                                                            "G1020",
                                                            "REFUGE",
                                                            "PAPILA"])

    #### rimone dataset
    with tab_rimone:
        example_img(path=PATH_RIMONE,
                    label=LABELS_USED)
    #### g1020 dataset
    with tab_g1020:
        example_img(path=PATH_G1020,
                    label=LABELS_USED)
    #### refuge dataset
    with tab_refuge:
        example_img(path=PATH_REFUGE,
                    label=LABELS_USED)
    #### papila dataset
    with tab_papila:
        example_img(path=PATH_PAPILA,
                    label=LABELS_USED)

## scenario info
with st.container():
    st.header("Skenario", anchor="info-skenario")
    st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM100}</div><br>",
                unsafe_allow_html=True)
    
    ## each scenario
    tab_s1, tab_s2, tab_s3 = st.tabs(["Skenario 1",
                                        "Skenario 2",
                                        "Skenario 3"])

## model result
with st.container():
    st.header("Performa Model", anchor="hasil-evaluasi")
    st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM100}</div><br>",
                unsafe_allow_html=True)

## summary and conclusion
with st.container():
    st.header("Kesimpulan", anchor="kesimpulan")
    st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM100}</div>",
                unsafe_allow_html=True)

## sidebar
st.sidebar.header("Judul", anchor="Info Penelitian")
st.sidebar.markdown(f'{THESIS_TITLE}')

st.sidebar.header("Penulis")
author_data = evaluation.load_df(path=PATH_AUTHOR_DATA)
for row in evaluation.load_df(path=PATH_AUTHOR_DATA).iterrows():
    with st.sidebar.expander(row[1][-1]):
        st.markdown(f"**{row[1][0]}**")
        st.text(f"{row[1][1]}")
        st.markdown(f"{row[1][2]}")

st.sidebar.header("Penelitian")
for path in [["Dataset", PATH_DATASET_DATA],
                ["Arsitektur", PATH_ARSITEKTUR_DATA]]:
    with st.sidebar.expander(path[0]):
        for row in evaluation.load_df(path=path[1]).iterrows():
            st.markdown(f"""**[{row[1][0]}]({row[1][1]} "{row[1][2]}")**""")