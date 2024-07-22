import streamlit as st
import pandas as pd
import hydralit_components as hc
import utils

st.set_page_config(layout="wide", page_title="Anime Data Analysis")

pages = {
    "Ranking Lists": "pages/1_Ranking_Lists.py",
    "Guided Data Exploration": "pages/2_Guided_Data_Exploration.py",
    "Self Data Exploration": "pages/3_Self_Data_Exploration.py",
    "Correlations": "pages/4_Correlations.py",
    "Simple Linear Regression": "pages/5_Simple_Linear_Regression.py",
    "Multiple Linear Regressions": "pages/6_Multiple_Linear_Regressions.py",
    "Advanced ML Algorithms": "pages/7_Advanced_ML_Algorithms.py",
    "Anime Popularity Predictor": "pages/8_Anime_Popularity_Predictor.py"
}

menu_data = [
    {'label': key, 'id': key} for key in pages.keys()
]

over_theme = {
    'txc_inactive': '#FFFFFF',
    'menu_background': '#333333',
    'txc_active': '#00C0F2'
}

menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    hide_streamlit_markers=True,
    sticky_nav=True,
    sticky_mode='sticky',
)

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"]{
        display: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)

if menu_id in pages:
    if menu_id != "Ranking Lists":
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"][aria-expanded="true"] {
                    display: none;
                }
                [data-testid="stSidebar"][aria-expanded="false"] {
                    display: none;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
                [data-testid="stSidebar"][aria-expanded="true"] {
                    display: block;
                }
                [data-testid="stSidebar"][aria-expanded="false"] {
                    display: block;
                }
            </style>
            """,
            unsafe_allow_html=True
        )
    utils.load_page(pages[menu_id])
else:

    st.markdown(
            """
            <style>
                [data-testid="stSidebar"][aria-expanded="true"] {
                    display: none;
                }
                [data-testid="stSidebar"][aria-expanded="false"] {
                    display: none;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

    @st.cache_data
    def load_data():
        return pd.read_csv("../data/animes_better.csv")

    data = load_data()

    st.title("Anime Data Analysis Project")

    st.header("Project Overview")
    st.write("""
    This project aims to analyze various aspects of anime using a comprehensive dataset. 
    Through this app, you can explore:

    - Rankings of anime based on different criteria
    - Guided visualizations of interesting trends and patterns
    - Interactive data exploration tools
    - Machine learning insights into what makes an anime popular
    - A predictor for estimating an anime's popularity based on its characteristics
    """)

    st.header("Dataset Preview")
    st.dataframe(data.head())

    st.header("Quick Stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Anime", len(data))
    with col2:
        st.metric("Average Score", round(data['score'].mean(), 2))
    with col3:
        st.metric("Most Common Genre", data['genre'].mode().values[0])