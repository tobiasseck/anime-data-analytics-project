import streamlit as st
import pandas as pd
import pygwalker as pyg
import streamlit.components.v1 as components

@st.cache_data
def load_data():
    df = pd.read_csv("../data/animes_better.csv")
    df['genre'] = df['genre'].apply(eval)
    df['year'] = pd.to_datetime(df['aired'], format='%Y-%m-%d', errors='coerce').dt.year
    return df

data = load_data()

st.title("Self Data Exploration")

st.write("""
Use this interactive interface to explore the anime dataset on your own. 
You can create various types of visualizations, filter data, and analyze trends just like you would in Tableau.

Instructions:
1. Drag and drop fields onto the rows, columns, or color areas to create visualizations.
2. Use the 'Chart type' dropdown to switch between different visualization types.
3. Apply filters by clicking on the funnel icon next to each field.
4. Explore relationships between different variables in the dataset.
""")

pyg_html = pyg.to_html(data)

components.html(pyg_html, height=1000, scrolling=True)