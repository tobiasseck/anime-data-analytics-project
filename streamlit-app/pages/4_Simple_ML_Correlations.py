import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from sklearn.preprocessing import MultiLabelBinarizer

st.set_page_config(layout="wide", page_title="Simple ML Correlations")

@st.cache_resource
def load_models_and_data():
    models = {}
    data = {}
    features = ['genre', 'Producers', 'Studios', 'Themes', 'Streaming Platforms', 
                'Duration', 'Type', 'Source', 'Demographic']
    for feature in features:
        feature_key = feature.lower().replace(" ", "_")
        with open(f'../models/simple_linear_regression_{feature_key}.pkl', 'rb') as f:
            models[feature] = pickle.load(f)
        with open(f'../data/simple_linear_regression_{feature_key}_data.pkl', 'rb') as f:
            data[feature] = pickle.load(f)
    return models, data

models, prepared_data = load_models_and_data()

st.title("Simple ML Correlations")

tabs = st.tabs([
    "Genre", "Producers", "Studios", "Themes", "Streaming Platforms",
    "Duration", "Type", "Source", "Demographic"
])

def display_correlation_results(feature, model, data):
    X = data[feature]['X']
    y = data[feature]['y']
    
    st.write(f"Correlation between {feature} and Popularity:")
    st.write(f"R-squared: {model.score(X, y):.4f}")
    st.write("Model Coefficients:")
    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    st.dataframe(coef_df.sort_values('Coefficient', ascending=False))

    #TODO:
    st.write("Visualizations will be added here.")

with tabs[0]:
    st.header("Genre vs Popularity")
    display_correlation_results("genre", models['genre'], prepared_data)

with tabs[1]:
    st.header("Producers vs Popularity")
    display_correlation_results("Producers", models['Producers'], prepared_data)

with tabs[2]:
    st.header("Studios vs Popularity")
    display_correlation_results("Studios", models['Studios'], prepared_data)

with tabs[3]:
    st.header("Themes vs Popularity")
    display_correlation_results("Themes", models['Themes'], prepared_data)

with tabs[4]:
    st.header("Streaming Platforms vs Popularity")
    display_correlation_results("Streaming Platforms", models['Streaming Platforms'], prepared_data)

with tabs[5]:
    st.header("Duration vs Popularity")
    display_correlation_results("Duration", models['Duration'], prepared_data)

with tabs[6]:
    st.header("Type vs Popularity")
    display_correlation_results("Type", models['Type'], prepared_data)

with tabs[7]:
    st.header("Source vs Popularity")
    display_correlation_results("Source", models['Source'], prepared_data)

with tabs[8]:
    st.header("Demographic vs Popularity")
    display_correlation_results("Demographic", models['Demographic'], prepared_data)