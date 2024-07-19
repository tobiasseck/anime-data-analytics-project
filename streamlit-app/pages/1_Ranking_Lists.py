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

@st.cache_data
def create_sorted_table(df, sort_column, ascending=False, limit=25):
    return df.sort_values(by=sort_column, ascending=ascending).head(limit)

@st.cache_data
def get_filter_options(df):
    all_genres = set([genre for genres in df['genre'] for genre in genres])
    return {
        'genres': ['All'] + sorted(all_genres),
        'demographics': ['All'] + df['Demographic'].dropna().unique().tolist(),
        'types': ['All'] + df['Type'].dropna().unique().tolist(),
        'sources': ['All'] + df['Source'].dropna().unique().tolist()
    }

st.title("Anime Rankings")

filter_options = get_filter_options(data)

# Sidebar filters
st.sidebar.header("Filters")

selected_genre = st.sidebar.selectbox("Select Genre", filter_options['genres'])
selected_demographic = st.sidebar.selectbox("Select Demographic", filter_options['demographics'])
selected_type = st.sidebar.selectbox("Select Type", filter_options['types'])
selected_source = st.sidebar.selectbox("Select Source", filter_options['sources'])

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
@st.cache_data
def apply_filters(df, filters):
    filtered = df.copy()
    if filters['genre'] != 'All':
        filtered = filtered[filtered['genre'].apply(lambda x: filters['genre'] in x)]
    if filters['demographic'] != 'All':
        filtered = filtered[filtered['Demographic'] == filters['demographic']]
    if filters['type'] != 'All':
        filtered = filtered[filtered['Type'] == filters['type']]
    if filters['source'] != 'All':
        filtered = filtered[filtered['Source'] == filters['source']]
    return filtered

filtered_data = apply_filters(data, st.session_state.filters)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Top by Rank", "Top by Score", "Top by Members", "Top by Popularity", "Top by Favorites"])

def display_anime_table(df, sort_column):
    for _, row in df.iterrows():
        col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 3, 2, 1, 1])
        
        with col1:
            try:
                response = requests.get(row['img_url'])
                img = Image.open(BytesIO(response.content))
                st.image(img, width=100)
            except:
                st.write("Image not available")
        
        with col2:
            st.subheader(row['title'])
        
        with col3:
            synopsis = row['synopsis'] if pd.notna(row['synopsis']) else "No synopsis available"
            st.write(synopsis[:200] + "..." if len(synopsis) > 200 else synopsis)
        
        with col4:
            producers = ", ".join(row['Producers']) if isinstance(row['Producers'], list) else "Unknown"
            st.write(f"Producers: {producers}")
        
        with col5:
            duration = row['Duration'] if pd.notna(row['Duration']) else "Unknown"
            st.metric("Duration", f"{int(duration)} min")
        
        with col6:
            value = row[sort_column] if pd.notna(row[sort_column]) else "N/A"
            st.metric(sort_column.capitalize(), value)
        
        st.markdown("---")

with tab1:
    st.header("Top 50 by Rank")
    ranked_table = create_sorted_table(filtered_data, 'ranked', ascending=True)
    display_anime_table(ranked_table, 'ranked')

with tab2:
    st.header("Top 50 by Score")
    score_table = create_sorted_table(filtered_data, 'score')
    display_anime_table(score_table, 'score')

with tab3:
    st.header("Top 50 by Members")
    members_table = create_sorted_table(filtered_data, 'members')
    display_anime_table(members_table, 'members')

with tab4:
    st.header("Top 50 by Popularity")
    popularity_table = create_sorted_table(filtered_data, 'popularity', ascending=True)
    display_anime_table(popularity_table, 'popularity')

with tab5:
    st.header("Top 50 by Favorites")
    favorites_table = create_sorted_table(filtered_data, 'Favorites')
    display_anime_table(favorites_table, 'Favorites')