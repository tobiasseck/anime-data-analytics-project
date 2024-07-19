import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MultiLabelBinarizer

st.set_page_config(layout="wide", page_title="Guided Data Exploration")

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: flex-end;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("../data/animes_better.csv")
    df['genre'] = df['genre'].apply(eval)
    df['year'] = pd.to_datetime(df['aired'], format='%Y-%m-%d', errors='coerce').dt.year
    return df

data = load_data()

st.title("Guided Data Exploration")

tabs = st.tabs([
    "Popularity vs Features", 
    "Popularity vs Categorical", 
    "Distributions", 
    "Genre Analysis", 
    "Time Trends", 
    "Correlations"
])

with tabs[0]:
    st.header("Popularity vs Numerical Features")
    features = ['episodes', 'ranked', 'score', 'members', 'Favorites', 'Duration']
    feature = st.selectbox("Select Feature", features)
    
    fig = px.scatter(data, x=feature, y='popularity', title=f'Popularity vs {feature}')
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.header("Popularity vs Categorical Features")
    categorical_features = ['Type', 'Source', 'Demographic']
    cat_feature = st.selectbox("Select Categorical Feature", categorical_features)
    
    fig = px.box(data, x=cat_feature, y='popularity', title=f'Popularity vs {cat_feature}')
    fig.update_layout(xaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("Distributions of Key Features")
    dist_features = ['members', 'popularity', 'ranked', 'score']
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=dist_features)
    
    for i, feature in enumerate(dist_features):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.add_trace(go.Histogram(x=data[feature].dropna(), name=feature), row=row, col=col)
    
    fig.update_layout(height=800, title_text="Distributions of Key Features")
    st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.header("Genre Analysis")
    
    all_genres = [genre for sublist in data['genre'] for genre in sublist]
    genre_counts = pd.Series(all_genres).value_counts()
    
    fig = px.bar(x=genre_counts.index[:20], y=genre_counts.values[:20], 
                 title='Top 20 Most Common Genres')
    st.plotly_chart(fig, use_container_width=True)
    
    genre_score_popularity = data.explode('genre').groupby('genre').agg({'score': 'mean', 'popularity': 'mean'}).sort_values(by='score', ascending=False)
    
    fig = px.bar(x=genre_score_popularity['score'][:10], y=genre_score_popularity.index[:10], 
                 title='Top 10 Genres by Average Score', orientation='h')
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.bar(x=genre_score_popularity['popularity'].sort_values()[:10], 
                 y=genre_score_popularity['popularity'].sort_values().index[:10], 
                 title='Top 10 Genres by Average Popularity', orientation='h')
    st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.header("Time Trends")

    data['year'] = data['aired'].str.extract(r'(\d{4})', expand=False)
    data['year'] = pd.to_numeric(data['year'], errors='coerce')

    data_with_year = data.dropna(subset=['year'])

    animes_per_year = data_with_year['year'].value_counts().sort_index().reset_index()
    animes_per_year.columns = ['Year', 'Count']
    fig = px.line(animes_per_year, x='Year', y='Count', 
                  title='Number of Animes Aired Per Year')
    st.plotly_chart(fig, use_container_width=True)

    all_genres = set([genre for sublist in data_with_year['genre'] for genre in sublist])
    popular_genres = ['Action', 'Comedy', 'Drama', 'Fantasy', 'Romance', 'School', 'Shounen']
    available_genres = [genre for genre in popular_genres if genre in all_genres]
    
    genre_trends = data_with_year.explode('genre')
    genre_trends = genre_trends[genre_trends['genre'].isin(available_genres)]
    genre_trends_over_time = genre_trends.groupby(['year', 'genre']).size().unstack(fill_value=0).reset_index()
    
    fig = px.line(genre_trends_over_time, x='year', y=available_genres, 
                  title='Trends of Selected Genres Over Time')
    st.plotly_chart(fig, use_container_width=True)

    average_score_per_year = data_with_year.groupby('year')['score'].mean().reset_index()
    fig = px.line(average_score_per_year, x='year', y='score', 
                  title='Average Anime Score Over Time')
    st.plotly_chart(fig, use_container_width=True)

    average_popularity_per_year = data_with_year.groupby('year')['popularity'].mean().reset_index()
    fig = px.line(average_popularity_per_year, x='year', y='popularity', 
                  title='Average Anime Popularity Over Time')
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.header("Feature Correlations with Popularity")

    @st.cache_data
    def get_correlations(df, column):
        if column == 'genre':
            mlb = MultiLabelBinarizer()
            feature_matrix = pd.DataFrame(mlb.fit_transform(df[column]), columns=mlb.classes_, index=df.index)
        else:
            feature_matrix = pd.get_dummies(df[column], prefix=column)
        
        feature_matrix['popularity'] = df['popularity']
        feature_corr = feature_matrix.corr()
        feature_corr_popularity = feature_corr['popularity'].drop('popularity').sort_values(key=abs, ascending=False)
        return feature_corr_popularity

    features = ['genre', 'Type', 'Source', 'Demographic', 'Themes']
    feature = st.selectbox("Select Feature", features)

    correlations = get_correlations(data, feature)
    
    fig = px.bar(
        x=correlations.head(20),
        y=correlations.head(20).index,
        orientation='h',
        title=f'Top 20 {feature.capitalize()} Correlations with Popularity',
        labels={'x': 'Correlation', 'y': feature.capitalize()}
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"Note: Negative correlation means the {feature.lower()} is associated with higher popularity (lower rank number).")

    st.write(f"Top 20 {feature.capitalize()} Correlations with Popularity:")
    st.write(correlations.head(20))