# pages/1_Ranking_Lists.py
import streamlit as st
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import ast

st.set_page_config(layout="wide", page_title="Anime Rankings")

@st.cache_data
def load_data():
    df = pd.read_csv("../data/animes_better.csv")
    df['genre'] = df['genre'].apply(ast.literal_eval)
    df['Producers'] = df['Producers'].apply(ast.literal_eval)
    return df

data = load_data()

st.title("Anime Rankings")

# Sidebar filters
st.sidebar.header("Filters")

# Genre filter
all_genres = set([genre for genres in data['genre'] for genre in genres])
genres = ['All'] + sorted(all_genres)
selected_genre = st.sidebar.selectbox("Select Genre", genres)

# Demographic filter
demographics = ['All'] + data['Demographic'].unique().tolist()
selected_demographic = st.sidebar.selectbox("Select Demographic", demographics)

# Type filter
types = ['All'] + data['Type'].unique().tolist()
selected_type = st.sidebar.selectbox("Select Type", types)

# Source filter
sources = ['All'] + data['Source'].unique().tolist()
selected_source = st.sidebar.selectbox("Select Source", sources)

col1, col2 = st.sidebar.columns(2)
apply_button = col1.button("Apply Filters")
reset_button = col2.button("Reset Filters")

if 'filters' not in st.session_state:
    st.session_state.filters = {
        'genre': 'All',
        'demographic': 'All',
        'type': 'All',
        'source': 'All'
    }

# Update filters when Apply is clicked
if apply_button:
    st.session_state.filters = {
        'genre': selected_genre,
        'demographic': selected_demographic,
        'type': selected_type,
        'source': selected_source
    }

# Reset filters when Reset is clicked
if reset_button:
    st.session_state.filters = {
        'genre': 'All',
        'demographic': 'All',
        'type': 'All',
        'source': 'All'
    }

# Apply filters
filtered_data = data.copy()
if st.session_state.filters['genre'] != 'All':
    filtered_data = filtered_data[filtered_data['genre'].apply(lambda x: st.session_state.filters['genre'] in x)]
if st.session_state.filters['demographic'] != 'All':
    filtered_data = filtered_data[filtered_data['Demographic'] == st.session_state.filters['demographic']]
if st.session_state.filters['type'] != 'All':
    filtered_data = filtered_data[filtered_data['Type'] == st.session_state.filters['type']]
if st.session_state.filters['source'] != 'All':
    filtered_data = filtered_data[filtered_data['Source'] == st.session_state.filters['source']]

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Top by Rank", "Top by Score", "Top by Members", "Top by Popularity", "Top by Favorites"])

def display_anime_table(df, sort_column, ascending=False, limit=50):
    sorted_df = df.sort_values(by=sort_column, ascending=ascending).head(limit)
    
    for _, row in sorted_df.iterrows():
        col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 3, 2, 1, 1])
        
        with col1:
            try:
                response = requests.get(row['img_url'])
                img = Image.open(BytesIO(response.content))
                st.image(img, width=100)
            except:
                st.write("Image not available")
        
        with col3:
            synopsis = row['synopsis'] if pd.notna(row['synopsis']) else "No synopsis available"
            st.write(synopsis[:200] + "..." if len(synopsis) > 200 else synopsis)
        
        with col4:
            producers = ", ".join(row['Producers']) if isinstance(row['Producers'], list) else "Unknown"
            st.write(f"Producers: {producers}")
        
        with col5:
            duration = row['Duration'] if pd.notna(row['Duration']) else "Unknown"
            st.metric("Duration", f"{duration} min")
        
        with col6:
            value = row[sort_column] if pd.notna(row[sort_column]) else "N/A"
            st.metric(sort_column.capitalize(), value)
        
        st.markdown("---")

with tab1:
    st.header("Top 50 by Rank")
    display_anime_table(filtered_data, 'ranked', ascending=True)

with tab2:
    st.header("Top 50 by Score")
    display_anime_table(filtered_data, 'score')

with tab3:
    st.header("Top 50 by Members")
    display_anime_table(filtered_data, 'members')

with tab4:
    st.header("Top 50 by Popularity")
    display_anime_table(filtered_data, 'popularity', ascending=True)

with tab5:
    st.header("Top 50 by Favorites")
    display_anime_table(filtered_data, 'Favorites')