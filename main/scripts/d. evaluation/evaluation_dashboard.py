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
PATH_SAMPLE = os.environ.get('PATH_DATA_AUG_SAMPLE')
PATH_EV_RESULT = os.environ.get('PATH_EVALUATION_RESULT')
### data path
PATH_AUTHOR_DATA = os.path.join(PATH_DATA, 'author_info.csv')
PATH_DATASET_DATA = os.path.join(PATH_DATA, 'dataset_info.csv')
PATH_ARSITEKTUR_DATA = os.path.join(PATH_DATA, 'arsitektur_info.csv')
PATH_SKENARIO_DATA = os.path.join(PATH_DATA, 'rfn_scenario_guide.csv')
PATH_AUG_IMG = os.path.join(PATH_DATA, 'example_aug_imgs.csv')
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
AUG_LIST = {'bright':'bright',
            'horizontal flip':'h_flip',
            'vertikal flip':'v_flip'}

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

def scenario_img(scenario:int,
                augment_type:dict):
    """showing the image for specific scenario and augmentation type

    Args:
        scenario (int): the scenario of the image
        augment_type (dict): a dictionary of the augmentation type
    """
    for aug_name, aug_type in augment_type.items():
        st.image(evaluation.img_scen_list(df_path=PATH_AUG_IMG,
                                        img_path=PATH_SAMPLE,
                                        scenario=scenario,
                                        augment_type=aug_type),
                caption=aug_name.title())

def show_df_result(file_name:str):
    """showing the dataframe result in the dashboard

    Args:
        file_name (str): the name of the file saved in the result folder
    """
    # import the dataframe
    df_eval = evaluation.load_df(path=os.path.join(PATH_EV_RESULT, f'{file_name}.csv'))
    
    # drop the unnecessary columns with ignore error
    df_eval.drop(columns=['true_positive', 'true_negative',
                            'false_positive', 'false_negative'],
                inplace=True, errors='ignore')
    # change the column name to uppercase
    df_columnns = [evaluation.string_upper(name) for name in df_eval.columns]
    df_eval.columns = df_columnns

    # show the dataframe
    st.dataframe(df_eval.style \
                    .highlight_max(axis=0, subset=df_eval.columns[-4:],
                                    props='color:black;background-color:#EE4540') \
                    .format(evaluation.string_upper),
                hide_index=True, use_container_width=True, height=175)

def show_val_result(name:str):
    """a template to show the evaluation result in the dashboard

    Args:
        name (str): the name of the scenario or dataset
    """
    # table result
    st.markdown(f'<div style="text-align: center; color: black"><b>Hasil Evaluasi Secara Umum</b></div>',
                unsafe_allow_html=True)
    show_df_result(f'{name}_general_result')
    st.markdown(f'<div style="text-align: center; color: black"><b>Hasil Evaluasi Akhir</b></div>',
                unsafe_allow_html=True)
    show_df_result(f'{name}_summarize_result')
    
    # confusion matrix
    st.markdown(f'<div style="text-align: center; color: black"><b>Confusion Matrix Model Terbaik</b></div>',
                unsafe_allow_html=True)
    st.image(os.path.join(PATH_EV_RESULT, f'confusion_matrix_{name}_general_best.png'),
            use_column_width=True)


# dasboard section
## summary and title
with st.container():
    st.title("Evaluasi Model Klasifikasi Glaukoma",
                anchor="judul-laporan")
    st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM100}</div>",
                unsafe_allow_html=True)

# ## info dataset
# with st.container():
#     ### dataset summary
#     dataset_info_df = evaluation.load_df(path=PATH_DATASET_DATA)
#     st.header("Dataset", anchor="info-dataset")
#     st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM10}</div><br>",
#                 unsafe_allow_html=True)
#     st.dataframe(dataset_info_df.loc[:, ~dataset_info_df.columns.isin(['Link', 'Sumber'])],
#                 hide_index=True, use_container_width=True)

#     ### example image
#     st.subheader('Contoh Data Citra', anchor=f'image-dataset')
#     tab_rimone, tab_g1020, tab_refuge, tab_papila = st.tabs(["RIM-ONE",
#                                                             "G1020",
#                                                             "REFUGE",
#                                                             "PAPILA"])

#     with tab_rimone:
#         example_img(path=PATH_RIMONE,
#                     label=LABELS_USED)
#     with tab_g1020:
#         example_img(path=PATH_G1020,
#                     label=LABELS_USED)
#     with tab_refuge:
#         example_img(path=PATH_REFUGE,
#                     label=LABELS_USED)
#     with tab_papila:
#         example_img(path=PATH_PAPILA,
#                     label=LABELS_USED)

# ## scenario info
# with st.container():
#     st.header("Skenario", anchor="info-scenario")
#     st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM100}</div><br>",
#                 unsafe_allow_html=True)
#     scenario_guide_df = evaluation.load_df(path=PATH_SKENARIO_DATA)
#     st.dataframe(scenario_guide_df.style.hide(),
#                 hide_index=True, use_container_width=True)
    
#     ## each scenario
#     st.subheader('Citra Hasil Augmentasi', anchor='image-scenario')
#     tab_s1, tab_s2, tab_s3 = st.tabs(["Skenario 1",
#                                         "Skenario 2",
#                                         "Skenario 3"])
    st.markdown("""
        <style>
            .stTabs [data-baseweb="tab-panel"]{
                background-color: white;
            }
            .st-emotion-cache-ltfnpr {
                color: black
            }
        </style>""", unsafe_allow_html=True)
    
    # with tab_s1:
    #     scenario_img(scenario=1,
    #                 augment_type={'no augmentation':'no'})
    # with tab_s2:
    #     scenario_img(scenario=2,
    #                 augment_type=AUG_LIST)
    # with tab_s3:
    #     scenario_img(scenario=3,
    #                 augment_type=AUG_LIST)


## model result
with st.container():
    st.header("Performa Model", anchor="hasil-evaluasi")

    st.subheader("Berdasarkan Skenario", anchor="skenario-evaluasi")
    tab_s1, tab_s2, tab_s3 = st.tabs(['Skenario 1',
                                        'Skenario 2',
                                        'Skenario 3'])
    
    with tab_s1:
        show_val_result('s1')
    with tab_s2:
        show_val_result('s2')
    with tab_s3:
        show_val_result('s3')

    st.subheader("Berdasarkan Dataset", anchor="dataset-evaluasi")
    tab_ev_rimone, tab_ev_g1020, tab_ev_refuge, tab_ev_papila = st.tabs(['Rim-One',
                                                                        'G1020',
                                                                        'REFUGE',
                                                                        'PAPILA'])
    
    with tab_ev_rimone:
        show_val_result('rimone')
    with tab_ev_g1020:
        show_val_result('g1020')
    with tab_ev_refuge:
        show_val_result('refuge')
    with tab_ev_papila:
        show_val_result('papila')
    
    st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM100}</div>",
                unsafe_allow_html=True)
    


## summary and conclusion
with st.container():
    st.header("Kesimpulan", anchor="kesimpulan")
    st.markdown(f"<div style='text-align: justify;'>{LOREM_IPSUM100}</div><br>",
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