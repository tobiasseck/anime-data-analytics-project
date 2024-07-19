import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objs as go
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf

st.set_page_config(layout="wide", page_title="Advanced ML Algorithms")

st.markdown("""<style>.stTabs [data-baseweb="tab-list"] {justify-content: flex-end;}</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    models = {}
    data = {}
    
    model_names = ['random_forest', 'random_forest_tuned', 'gradient_boosting', 'xgboost']
    for model in model_names:
        model_path = f'./models/{model}.pkl'
        data_path = f'./data/prepared_{model}_data.pkl'
        
        try:
            with open(model_path, 'rb') as f:
                models[model] = pickle.load(f)
            with open(data_path, 'rb') as f:
                data[model] = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load {model} model or data: {str(e)}")
    
    nn_model_path = './models/neural_network_savedmodel'
    nn_data_path = './data/prepared_neural_network_data.pkl'
    
    try:
        models['neural_network'] = tf.saved_model.load(nn_model_path)
        with open(nn_data_path, 'rb') as f:
            data['neural_network'] = pickle.load(f)
    except Exception as e:
        st.warning(f"Failed to load neural network model or data: {str(e)}")
    
    return models, data

models, prepared_data = load_models_and_data()

def create_prediction_plot(y_true, y_pred, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', marker=dict(color='blue', opacity=0.5),
                             name='Predictions'))
    fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()],
                             mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
    fig.update_layout(title=title, xaxis_title='Actual Popularity', yaxis_title='Predicted Popularity', height=600,  autosize=True)
    return fig

def create_residual_plot(y_true, y_pred, title):
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', marker=dict(color='blue', opacity=0.5)))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis_title='Predicted Popularity', yaxis_title='Residuals', height=600,  autosize=True)
    return fig

def display_model_results(model_name, model, data):
    
    if isinstance(data, dict):
        if 'X_test' in data and 'y_test' in data:
            X_test, y_test = data['X_test'], data['y_test']
        elif 'X' in data and 'y' in data:
            X_test, y_test = data['X'], data['y']
        else:
            st.error(f"Unexpected data structure for {model_name}. Please check the pickled data.")
            return
    elif isinstance(data, tuple) and len(data) == 2:
        X_test, y_test = data
    else:
        st.error(f"Unexpected data structure for {model_name}. Please check the pickled data.")
        return

    try:
        if model_name == 'neural_network':
            serving_default = model.signatures['serving_default']
            input_tensor_name = list(serving_default.structured_input_signature[1].keys())[0]
            output_tensor_name = list(serving_default.structured_outputs.keys())[0]
            y_pred = serving_default(**{input_tensor_name: tf.convert_to_tensor(X_test.astype('float32'))})[output_tensor_name].numpy().flatten()
        else:
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r2_model_score = model.score(X_test, y_test) if hasattr(model, 'score') else "N/A"
        
        r2_model_score_str = f"{r2_model_score:.2f}" if isinstance(r2_model_score, float) else r2_model_score
        
        st.write(f"{model_name} - MSE: {mse:.2f}, R² (sklearn): {r2:.2f}, R² (model.score): {r2_model_score_str}")
        
        fig1 = create_prediction_plot(y_test, y_pred, f'Actual vs Predicted Popularity ({model_name})')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = create_residual_plot(y_test, y_pred, f'Residuals Distribution for {model_name}')
        st.plotly_chart(fig2, use_container_width=True)
        
        if model_name == 'xgboost':
            display_xgboost_feature_importance(model, X_test)
        
        if model_name == 'neural_network':
            display_neural_network_history(data)
        
    except Exception as e:
        st.error(f"An error occurred while making predictions: {str(e)}")
        st.exception(e)
        return

def display_xgboost_feature_importance(model, X_test):
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Category': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig3 = px.bar(feature_importance.head(20), x='Importance', y='Category', orientation='h',
                      title='Top 20 Feature Importances (XGBoost)')
        fig3.update_layout(
            height=800, 
            width=None, 
            autosize=True,
            yaxis={'categoryorder':'total ascending'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.write("Feature importances not available for this model.")

def display_neural_network_history(data):
    if isinstance(data, dict) and 'history' in data:
        history = data['history']
        if isinstance(history, dict) and 'loss' in history and 'val_loss' in history:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(y=history['loss'], name='Training Loss'))
            fig4.add_trace(go.Scatter(y=history['val_loss'], name='Validation Loss'))
            fig4.update_layout(title='Learning Curves', xaxis_title='Epoch', yaxis_title='Mean Squared Error')
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.write("Learning curves data not found in history.")
    else:
        st.write("Learning curves not available for this model.")        

st.title("Advanced ML Algorithms")

tabs = st.tabs(["Random Forest", "Random Forest (Tuned)", "Gradient Boosting", "Neural Network", "XGBoost"])

with tabs[0]:
    st.header("Random Forest")
    if 'random_forest' in models and 'random_forest' in prepared_data:
        display_model_results('random_forest', models['random_forest'], prepared_data['random_forest'])
    else:
        st.write("Random Forest model or data not available.")

with tabs[1]:
    st.header("Random Forest (Tuned)")
    if 'random_forest_tuned' in models and 'random_forest_tuned' in prepared_data:
        display_model_results('random_forest_tuned', models['random_forest_tuned'], prepared_data['random_forest_tuned'])
    else:
        st.write("Tuned Random Forest model or data not available.")

with tabs[2]:
    st.header("Gradient Boosting")
    if 'gradient_boosting' in models and 'gradient_boosting' in prepared_data:
        display_model_results('gradient_boosting', models['gradient_boosting'], prepared_data['gradient_boosting'])
    else:
        st.write("Gradient Boosting model or data not available.")

with tabs[3]:
    st.header("Neural Network")
    if 'neural_network' in models and 'neural_network' in prepared_data:
        display_model_results('neural_network', models['neural_network'], prepared_data['neural_network'])
    else:
        st.write("Neural Network model or data not available.")

with tabs[4]:
    st.header("XGBoost")
    if 'xgboost' in models and 'xgboost' in prepared_data:
        display_model_results('xgboost', models['xgboost'], prepared_data['xgboost'])
    else:
        st.write("XGBoost model or data not available.")