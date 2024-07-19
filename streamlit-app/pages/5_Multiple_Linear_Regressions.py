import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Multiple Linear Regressions")

@st.cache_resource
def load_models_and_data():
    models = {}
    data = {}
    regression_types = [
        'genre_platform',
        'producer_genre_platform',
        'type_source_demographic_producer_genre_platform'
    ]
    for reg_type in regression_types:
        with open(f'../models/multiple_regression_{reg_type}.pkl', 'rb') as f:
            models[reg_type] = pickle.load(f)
        with open(f'../data/prepared_{reg_type}_data.pkl', 'rb') as f:
            data[reg_type] = pickle.load(f)
    return models, data

models, prepared_data = load_models_and_data()

st.title("Multiple Linear Regressions")

tabs = st.tabs([
    "Genre and Platform",
    "Producer, Genre, and Platform",
    "Type, Source, Demographic, Producer, Genre, and Platform"
])

def display_multiple_regression_results(reg_type, model, data):
    X = data[reg_type]['X']
    y = data[reg_type]['y']
    
    st.write(f"Multiple Linear Regression: {reg_type.replace('_', ', ').title()} vs Popularity")
    st.write(f"R-squared: {model.score(X, y):.4f}")

    coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    coef_df = coef_df.sort_values('Coefficient', ascending=False)
    st.write("Top 10 Model Coefficients:")
    st.dataframe(coef_df.head(10))

    # Predictions
    y_pred = model.predict(X)

    # Actual vs Predicted Plot
    fig1 = px.scatter(x=y, y=y_pred, labels={'x': 'Actual Popularity', 'y': 'Predicted Popularity'},
                      title=f'Actual vs Predicted Popularity')
    fig1.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()],
                              mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig1)

    # Residual Plot
    residuals = y - y_pred
    fig2 = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Popularity', 'y': 'Residuals'},
                      title=f'Residuals Distribution')
    fig2.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig2)

    # Top and Bottom Coefficients
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=('Top 20 Coefficients', 'Bottom 20 Coefficients'))
    
    # Top 20 Coefficients
    top_20 = coef_df.head(20)
    fig3.add_trace(go.Bar(x=top_20['Coefficient'], y=top_20['Feature'], orientation='h', name='Top 20'), row=1, col=1)
    
    # Bottom 20 Coefficients
    bottom_20 = coef_df.tail(20)
    fig3.add_trace(go.Bar(x=bottom_20['Coefficient'], y=bottom_20['Feature'], orientation='h', name='Bottom 20'), row=1, col=2)
    
    fig3.update_layout(height=600, title_text='Top and Bottom 20 Coefficients')
    st.plotly_chart(fig3)

with tabs[0]:
    display_multiple_regression_results('genre_platform', models['genre_platform'], prepared_data)

with tabs[1]:
    display_multiple_regression_results('producer_genre_platform', models['producer_genre_platform'], prepared_data)

with tabs[2]:
    display_multiple_regression_results('type_source_demographic_producer_genre_platform', 
                                        models['type_source_demographic_producer_genre_platform'], 
                                        prepared_data)