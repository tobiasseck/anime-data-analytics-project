import streamlit as st
import pickle
import os
import tensorflow as tf

@st.cache_resource
def load_models_and_data():
    models = {}
    data = {}

    base_path = os.path.dirname(os.path.abspath(__file__))
    
    regression_types = [
        'genre_platform',
        'producer_genre_platform',
        'type_source_demographic_producer_genre_platform'
    ]
    
    for reg_type in regression_types:
        model_path = os.path.join(base_path, 'models', f'multiple_regression_{reg_type}.pkl')
        data_path = os.path.join(base_path, 'data', f'prepared_multiple_regression_{reg_type}_data.pkl')
        
        try:
            with open(model_path, 'rb') as f:
                models[f'multiple_regression_{reg_type}'] = pickle.load(f)
            with open(data_path, 'rb') as f:
                data[f'multiple_regression_{reg_type}'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load multiple_regression_{reg_type} model or data: {str(e)}")

    other_models = ['random_forest', 'random_forest_tuned', 'gradient_boosting', 'xgboost']
    for model in other_models:
        model_path = os.path.join(base_path, 'models', f'{model}.pkl')
        data_path = os.path.join(base_path, 'data', f'prepared_{model}_data.pkl')
        
        try:
            with open(model_path, 'rb') as f:
                models[model] = pickle.load(f)
            with open(data_path, 'rb') as f:
                data[model] = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load {model} model or data: {str(e)}")

    nn_model_path = os.path.join(base_path, 'models', 'neural_network_savedmodel')
    nn_data_path = os.path.join(base_path, 'data', 'prepared_neural_network_data.pkl')
    
    try:
        models['neural_network'] = tf.saved_model.load(nn_model_path)
        with open(nn_data_path, 'rb') as f:
            data['neural_network'] = pickle.load(f)
    except Exception as e:
        st.warning(f"Failed to load neural network model or data: {str(e)}")
    
    return models, data