import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: flex-end;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    models = {}
    data = {}
    features = ['genre', 'Producers', 'Studios', 'Themes', 'Streaming Platforms', 
                'Duration', 'Type', 'Source', 'Demographic']
    for feature in features:
        feature_key = feature.lower().replace(" ", "_")
        with open(f'./models/simple_linear_regression_{feature_key}.pkl', 'rb') as f:
            models[feature] = pickle.load(f)
        with open(f'./data/simple_linear_regression_{feature_key}_data.pkl', 'rb') as f:
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
    
    st.write(f"#### Correlation between {feature} and Popularity --> **R-squared: {model.score(X, y):.4f}**")

    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    elif isinstance(X, np.ndarray) and X.ndim == 2:
        feature_names = [f"{feature}_{i}" for i in range(X.shape[1])]
    else:
        feature_names = [feature]

    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_.flatten()})
    coef_df = coef_df.sort_values('Coefficient', ascending=False)
    
    y_pred = model.predict(X)

    fig1 = px.scatter(x=y, y=y_pred, labels={'x': 'Actual Popularity', 'y': 'Predicted Popularity'},
                      title=f'Actual vs Predicted Popularity for {feature}')
    fig1.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()],
                              mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig1)

    residuals = y - y_pred
    fig2 = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted Popularity', 'y': 'Residuals'},
                      title=f'Residuals Distribution for {feature}')
    fig2.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig2)

    if feature in ['genre', 'Type', 'Source', 'Demographic']:
        fig3 = make_subplots(rows=1, cols=2, subplot_titles=('Top 20 Coefficients', 'Bottom 20 Coefficients'))

        top_20 = coef_df.head(20)
        fig3.add_trace(go.Bar(x=top_20['Coefficient'], y=top_20['Feature'], orientation='h', name='Top 20'), row=1, col=1)

        bottom_20 = coef_df.tail(20)
        fig3.add_trace(go.Bar(x=bottom_20['Coefficient'], y=bottom_20['Feature'], orientation='h', name='Bottom 20'), row=1, col=2)
        
        fig3.update_layout(height=600, title_text=f'Top and Bottom 20 Coefficients for {feature}')
        st.plotly_chart(fig3)

    st.write("Model Coefficients:")
    st.dataframe(coef_df)

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