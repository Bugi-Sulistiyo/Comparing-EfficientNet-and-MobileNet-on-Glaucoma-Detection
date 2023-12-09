import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.environ.get('PATH_CUSTOM_MODULES'))

import random

import streamlit as st

import data_handling_dashboard as dh

# global variable
## path
PATH_DATA = os.environ.get('PATH_DATA_DASHBOARD')
PATH_AUTHOR_DATA = os.path.join(PATH_DATA, 'author_info.csv')
PATH_DATASET_DATA = os.path.join(PATH_DATA, 'dataset_info.csv')
PATH_ARSITEKTUR_DATA = os.path.join(PATH_DATA, 'arsitektur_info.csv')
PATH_EVALUASTION_DATA = os.path.join(PATH_DATA, 'dummy_evaluation_model.xlsx')
PATH_DATASET = os.environ.get('PATH_DATASET_COMBINED')
PATH_RIMONE = os.path.join(PATH_DATASET, 'rimone')
PATH_G1020 = os.path.join(PATH_DATASET, 'g1020')
PATH_REFUGE = os.path.join(PATH_DATASET, 'refuge')
PATH_PAPILA = os.path.join(PATH_DATASET, 'papila')
## others
LOREM_IPSUM100 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec vitae lobortis arcu. Quisque pulvinar venenatis libero, vitae semper metus euismod pulvinar. Nam justo ligula, laoreet ut malesuada et, egestas eu quam. Quisque pharetra bibendum ultricies. Etiam condimentum risus enim. Mauris semper, mauris sed aliquet blandit, purus tellus porta velit, vitae auctor magna magna sed neque. Nunc vestibulum tincidunt ligula, ut scelerisque mi molestie non. Aenean nec maximus arcu. Integer gravida tellus sit amet rhoncus consequat. Integer scelerisque placerat felis at dignissim. Sed eget auctor sem, rhoncus condimentum odio. Nullam sem dolor, mollis id ultrices sit amet, sollicitudin nec nulla. Vivamus."
LOREM_IPSUM10 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec vitae lobortis arcu"
THESIS_TITLE = "Perbandingan Arsitektur MobileNet dan EfficientNet pada Klasifikasi Glaukoma Berdasarkan Citra Fundus"
LABELS_USED = ["Normal", "Glaukoma"]

# function avoid redundant code
def show_example_image(path:str, label:str):
    """show example image from dataset

    Args:
        path (str): path where the image is stored
        label (str): which label to show
    """
    g_norm_imgs_name = dh.get_image_path(os.path.join(path,
                                                    label))
    g_norm_imgs_array = dh.load_image_array(random.choice(g_norm_imgs_name))
    st.image(g_norm_imgs_array,
            caption=label,
            use_column_width=True)

def show_general_evaluation_result(path:str,
                                    x:str,
                                    y_train_val:list,
                                    y_test:list,
                                    title_train_val:str,
                                    title_test:str,
                                    x_axis:str,
                                    y_axis:str,
                                    text_train_val:str,
                                    text_test:str):
    """show general evaluation result which is containing accuracy and loss plot.

    Args:
        path (str): path to evaluation result in excel or csv format
        x (str): the column name for x axis
        y_train_val (list): the column name for y axis in training and validation
        y_test (list): the column name for y axis in testing
        title_train_val (str): title for training and validation plot
        title_test (str): title for testing plot
        x_axis (str): the label for x axis
        y_axis (str): the label for y axis
        text_train_val (str): description text to show in training and validation plot
        text_test (str): description text to show in testing plot
    """
    st.pyplot(dh.line_chart(dh.load_dataframe(path),
                            x=x,
                            y=y_train_val,
                            title=title_train_val,
                            x_axis=x_axis,
                            y_axis=y_axis))
    st.markdown(f"<div style='text-align: justify;'>{text_train_val}</div>",
                unsafe_allow_html=True)
    
    st.pyplot(dh.line_chart(dh.load_dataframe(path),
                            x=x,
                            y=y_test,
                            title=title_test,
                            x_axis=x_axis,
                            y_axis=y_axis))
    st.markdown(f"<div style='text-align: justify;'>{text_test}</div>",
                unsafe_allow_html=True)

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

    ### rimone dataset
    with tab_rimone:
        with st.container():
            #### showing example image
            st.subheader("Contoh Gambar", anchor="rimone-dataset")

            col_normal_image, col_glaucoma_image = st.columns(2)
            with col_normal_image:
                show_example_image(PATH_RIMONE, LABELS_USED[0])
            with col_glaucoma_image:
                show_example_image(PATH_RIMONE, LABELS_USED[1])
        
        with st.container():
            #### showing accuracy and loss plot
            st.subheader("Grafik Akurasi dan Loss", anchor="rimone-acc-loss")
            col_acc, col_loss = st.columns(2)
            with col_acc:
                show_general_evaluation_result(path=PATH_EVALUASTION_DATA,
                                                x="epoch",
                                                y_train_val=["Akurasi Train", "Akurasi Val"],
                                                y_test=["Akurasi Test"],
                                                title_train_val="Evaluasi Model Pada Data Latih dan Validasi",
                                                title_test="Evaluasi Model Pada Data Uji",
                                                x_axis="Epoch", y_axis="Akurasi",
                                                text_train_val=LOREM_IPSUM10,
                                                text_test=LOREM_IPSUM10)
            with col_loss:
                show_general_evaluation_result(path=PATH_EVALUASTION_DATA,
                                                x="epoch",
                                                y_train_val=["Loss Train", "Loss Val"],
                                                y_test=["Loss Test"],
                                                title_train_val="Evaluasi Model Pada Data Latih dan Validasi",
                                                title_test="Evaluasi Model Pada Data Uji",
                                                x_axis="Epoch", y_axis="Loss",
                                                text_train_val=LOREM_IPSUM10,
                                                text_test=LOREM_IPSUM10)
            
    
    ### g1020 dataset
    with tab_g1020:
        with st.container():
            #### showing example image
            st.subheader("Contoh Gambar", anchor="g1020-dataset")

            col_normal_image, col_glaucoma_image = st.columns(2)
            with col_normal_image:
                show_example_image(PATH_G1020, LABELS_USED[0])
            with col_glaucoma_image:
                show_example_image(PATH_G1020, LABELS_USED[1])
        
        with st.container():
            #### showing accuracy and loss plot
            st.subheader("Grafik Akurasi dan Loss", anchor="g1020-acc-loss")
            col_acc, col_loss = st.columns(2)
            with col_acc:
                show_general_evaluation_result(path=PATH_EVALUASTION_DATA,
                                                x="epoch",
                                                y_train_val=["Akurasi Train", "Akurasi Val"],
                                                y_test=["Akurasi Test"],
                                                title_train_val="Evaluasi Model Pada Data Latih dan Validasi",
                                                title_test="Evaluasi Model Pada Data Uji",
                                                x_axis="Epoch", y_axis="Akurasi",
                                                text_train_val=LOREM_IPSUM10,
                                                text_test=LOREM_IPSUM10)
            with col_loss:
                show_general_evaluation_result(path=PATH_EVALUASTION_DATA,
                                                x="epoch",
                                                y_train_val=["Loss Train", "Loss Val"],
                                                y_test=["Loss Test"],
                                                title_train_val="Evaluasi Model Pada Data Latih dan Validasi",
                                                title_test="Evaluasi Model Pada Data Uji",
                                                x_axis="Epoch", y_axis="Loss",
                                                text_train_val=LOREM_IPSUM10,
                                                text_test=LOREM_IPSUM10)

    ### refuge dataset
    with tab_refuge:
        with st.container():
            #### showing example image
            st.subheader("Contoh Gambar", anchor="refuge-dataset")

            col_normal_image, col_glaucoma_image = st.columns(2)
            with col_normal_image:
                show_example_image(PATH_REFUGE, LABELS_USED[0])
            with col_glaucoma_image:
                show_example_image(PATH_REFUGE, LABELS_USED[1])
        
        with st.container():
            #### showing accuracy and loss plot
            st.subheader("Grafik Akurasi dan Loss", anchor="refuge-acc-loss")
            col_acc, col_loss = st.columns(2)
            with col_acc:
                show_general_evaluation_result(path=PATH_EVALUASTION_DATA,
                                                x="epoch",
                                                y_train_val=["Akurasi Train", "Akurasi Val"],
                                                y_test=["Akurasi Test"],
                                                title_train_val="Evaluasi Model Pada Data Latih dan Validasi",
                                                title_test="Evaluasi Model Pada Data Uji",
                                                x_axis="Epoch", y_axis="Akurasi",
                                                text_train_val=LOREM_IPSUM10,
                                                text_test=LOREM_IPSUM10)
            with col_loss:
                show_general_evaluation_result(path=PATH_EVALUASTION_DATA,
                                                x="epoch",
                                                y_train_val=["Loss Train", "Loss Val"],
                                                y_test=["Loss Test"],
                                                title_train_val="Evaluasi Model Pada Data Latih dan Validasi",
                                                title_test="Evaluasi Model Pada Data Uji",
                                                x_axis="Epoch", y_axis="Loss",
                                                text_train_val=LOREM_IPSUM10,
                                                text_test=LOREM_IPSUM10)

    ### papila dataset
    with tab_papila:
        with st.container():
            #### showing example image
            st.subheader("Contoh Gambar", anchor="papila-dataset")

            col_normal_image, col_glaucoma_image = st.columns(2)
            with col_normal_image:
                show_example_image(PATH_PAPILA, LABELS_USED[0])
            with col_glaucoma_image:
                show_example_image(PATH_PAPILA, LABELS_USED[1])
        
        with st.container():
            #### showing accuracy and loss plot
            st.subheader("Grafik Akurasi dan Loss", anchor="papila-acc-loss")
            col_acc, col_loss = st.columns(2)
            with col_acc:
                show_general_evaluation_result(path=PATH_EVALUASTION_DATA,
                                                x="epoch",
                                                y_train_val=["Akurasi Train", "Akurasi Val"],
                                                y_test=["Akurasi Test"],
                                                title_train_val="Evaluasi Model Pada Data Latih dan Validasi",
                                                title_test="Evaluasi Model Pada Data Uji",
                                                x_axis="Epoch", y_axis="Akurasi",
                                                text_train_val=LOREM_IPSUM10,
                                                text_test=LOREM_IPSUM10)
            with col_loss:
                show_general_evaluation_result(path=PATH_EVALUASTION_DATA,
                                                x="epoch",
                                                y_train_val=["Loss Train", "Loss Val"],
                                                y_test=["Loss Test"],
                                                title_train_val="Evaluasi Model Pada Data Latih dan Validasi",
                                                title_test="Evaluasi Model Pada Data Uji",
                                                x_axis="Epoch", y_axis="Loss",
                                                text_train_val=LOREM_IPSUM10,
                                                text_test=LOREM_IPSUM10)


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