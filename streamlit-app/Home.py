import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide", page_title="Anime Data Analysis")

@st.cache_data
def load_data():
    return pd.read_csv("../data/animes_better.csv")

data = load_data()

st.title("Anime Data Analysis Project")
st.write("Welcome to my Streamlit app showcasing an analysis of anime data!")

st.header("Project Overview")
st.write("""
This project aims to analyze various aspects of anime using a comprehensive dataset. 
Through this app, you can explore:

- Rankings of anime based on different criteria
- Guided visualizations of interesting trends and patterns
- Interactive data exploration tools
- Machine learning insights into what makes an anime popular
- A predictor for estimating an anime's popularity based on its characteristics

Use the sidebar to navigate through different sections of the app.
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