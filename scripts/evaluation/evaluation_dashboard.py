import numpy as np
import pandas as pd

import streamlit as st
import matplotlib.pyplot as plt

# global variable
THESIS_TITLE = "Perbandingan Arsitektur MobileNet dan EfficientNet pada Klasifikasi Glaukoma Berdasarkan Citra Fundus"

# dasboard section
## summary and title
with st.container():
    st.title("Laporan Evaluasi Model Klasifikasi Glaukoma",
             anchor="judul-laporan")

## sidebar
st.sidebar.header("Judul")
st.sidebar.text(THESIS_TITLE)

st.sidebar.header("Penulis")
for author_type, author_info in AUTHOR.items():
    with st.sidebar.expander(author_type):
        for info_type, info_value in author_info.items():
            st.markdown(f"""**{info_type}**\n
                        {info_value}""")

st.sidebar.header("Penelitian")