import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import ast

st.set_page_config(layout="wide", page_title="Feature Correlations")

st.markdown("""<style>.stTabs [data-baseweb="tab-list"] {justify-content: flex-end;}</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("../data/animes_better.csv")
    df['genre'] = df['genre'].apply(ast.literal_eval)
    return df

data = load_data()

st.title("Feature Correlations")

def create_correlation_heatmap(corr_matrix, title):
    if isinstance(corr_matrix, pd.Series):
        corr_matrix = corr_matrix.to_frame()
    
    fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmin=-1, zmax=1,
                    colorbar=dict(title='Correlation')
                    ))

    fig.update_layout(
        title=title,
        xaxis_tickangle=-45,
        width=None,
        height=800,
        autosize=True
    )
    return fig

def highlight_popularity(x):
    return ['background-color: yellow' if 'popularity' in i else '' for i in x.index]

tabs = st.tabs(["Full Heatmap", "Numeric Values", "Genres vs Popularity", 
                "Demographics vs Popularity", "Source vs Popularity", "Type vs Popularity"])

with tabs[0]:
    st.header("Full Correlation Heatmap")
    
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = ['Type', 'Source', 'Demographic']
    
    genre_dummies = pd.DataFrame({genre: data['genre'].apply(lambda x: genre in x).astype(int) for genre in set([item for sublist in data['genre'] for item in sublist])})
    
    correlation_data = pd.concat([data[numerical_cols + categorical_cols], genre_dummies], axis=1)
    
    for col in categorical_cols:
        dummies = pd.get_dummies(correlation_data[col], prefix=col)
        correlation_data = pd.concat([correlation_data, dummies], axis=1)
        correlation_data.drop(col, axis=1, inplace=True)

    correlation_matrix = correlation_data.corr()
    
    fig = create_correlation_heatmap(correlation_matrix, 'Full Correlation Heatmap of Anime Features')
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("Correlation Values:")
    st.dataframe(correlation_matrix.style.apply(highlight_popularity, axis=1))

with tabs[1]:
    st.header("Numeric Values Correlation")
    numeric_corr = data[numerical_cols].corr()
    fig = go.Figure(data=go.Heatmap(
                    z=numeric_corr.values,
                    x=numeric_corr.columns,
                    y=numeric_corr.index,
                    colorscale='RdBu',
                    zmin=-1, zmax=1,
                    colorbar=dict(title='Correlation'),
                    text=numeric_corr.values.round(2),
                    texttemplate="%{text}",
                    textfont={"size":14}
                    ))
    fig.update_layout(
        title='Correlation Heatmap of Numeric Features',
        xaxis_tickangle=0,
        width=None,
        height=800,
        autosize=True
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(numeric_corr.style.apply(highlight_popularity, axis=1))

def create_bar_chart(data, title):
    df = data.reset_index()
    df.columns = ['Category', 'Correlation']
    fig = px.bar(df, x='Correlation', y='Category', orientation='h',
                 labels={'Correlation':'Correlation', 'Category':''}, title=title)
    fig.update_layout(
        height=800, 
        width=None, 
        autosize=True,
        yaxis={'categoryorder':'total ascending'},
        xaxis_title="Correlation (Negative values indicate higher popularity)",
    )
    fig.add_vline(x=0, line_dash="dot", line_color="white")
    return fig

with tabs[2]:
    st.header("Genres vs Popularity")
    genres_pop_corr = pd.concat([genre_dummies, data['popularity']], axis=1).corr()
    genres_pop_corr = genres_pop_corr.drop('popularity', axis=0)['popularity']
    fig = create_bar_chart(genres_pop_corr, 'Genres vs Popularity Correlation')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(genres_pop_corr)

with tabs[3]:
    st.header("Demographics vs Popularity")
    demo_pop_corr = pd.get_dummies(data['Demographic']).corrwith(data['popularity']).sort_values(ascending=True)
    fig = create_bar_chart(demo_pop_corr, 'Demographics vs Popularity Correlation')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(demo_pop_corr)

with tabs[4]:
    st.header("Source vs Popularity")
    source_pop_corr = pd.get_dummies(data['Source']).corrwith(data['popularity']).sort_values(ascending=True)
    fig = create_bar_chart(source_pop_corr, 'Source vs Popularity Correlation')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(source_pop_corr)

with tabs[5]:
    st.header("Type vs Popularity")
    type_pop_corr = pd.get_dummies(data['Type']).corrwith(data['popularity']).sort_values(ascending=True)
    fig = create_bar_chart(type_pop_corr, 'Type vs Popularity Correlation')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(type_pop_corr)

csv = correlation_matrix.to_csv(index=True)
st.download_button(
    label="Download Full Correlation Matrix as CSV",
    data=csv,
    file_name="anime_feature_correlations.csv",
    mime="text/csv",
)