import pandas as pd
import streamlit as st
import pickle
import os
import tensorflow as tf
import numpy as np
import pickle
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import itertools
import time

from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf

from plotly.subplots import make_subplots

import ast

import pygwalker as pyg
import streamlit.components.v1 as components

from sklearn.preprocessing import MultiLabelBinarizer

from PIL import Image
import requests
from io import BytesIO


def load_page(page_path):
    with open(page_path, 'r') as file:
        code = file.read()
        try:
            exec_globals = globals().copy()
            exec_globals.update({
                'st': st,
                'pd': pd,
                'np': np,
                'go': go,
                'px': px,
                'tf': tf,
                'r2_score': r2_score,
                'mean_squared_error': mean_squared_error,
                'load_models_and_data': load_models_and_data,
                'session_state': st.session_state,
                'pickle': pickle,
                'stats': stats,
                'itertools': itertools,
                'time': time,
                'make_subplots': make_subplots,
                'ast': ast,
                'pyg': pyg,
                'components': components,
                'MultiLabelBinarizer': MultiLabelBinarizer,
                'Image': Image,
                'requests': requests,
                'BytesIO': BytesIO
            })
            exec(code, exec_globals)
        except Exception as e:
            st.error(f"Fehläääär auf {page_path}: {e}")

@st.cache_resource
def load_models_and_data():
    models = {}
    data = {}
    total_anime = 0

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
                loaded_data = pickle.load(f)
                data[f'multiple_regression_{reg_type}'] = {
                    'feature_names': loaded_data.get('feature_names', []),
                    'genre_encoder': loaded_data.get('genre_encoder'),
                    'type_encoder': loaded_data.get('type_encoder'),
                    'source_encoder': loaded_data.get('source_encoder'),
                    'demographic_encoder': loaded_data.get('demographic_encoder'),
                    'scaler': loaded_data.get('scaler'),
                    'original_data': loaded_data
                }
                if 'X' in loaded_data and isinstance(loaded_data['X'], pd.DataFrame):
                    total_anime = max(total_anime, loaded_data['X'].shape[0])                 
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
                loaded_data = pickle.load(f)
                data[model] = {
                    'feature_names': loaded_data.get('feature_names', []),
                    'genre_encoder': loaded_data.get('genre_encoder'),
                    'type_encoder': loaded_data.get('type_encoder'),
                    'source_encoder': loaded_data.get('source_encoder'),
                    'demographic_encoder': loaded_data.get('demographic_encoder'),
                    'scaler': loaded_data.get('scaler'),
                    'original_data': loaded_data
                }
                if 'X' in loaded_data and isinstance(loaded_data['X'], pd.DataFrame):
                    total_anime = max(total_anime, loaded_data['X'].shape[0])
        except Exception as e:
            st.warning(f"Failed to load {model} model or data: {str(e)}")

    nn_model_path = os.path.join(base_path, 'models', 'neural_network_savedmodel')
    nn_data_path = os.path.join(base_path, 'data', 'prepared_neural_network_data.pkl')
    
    try:
        models['neural_network'] = tf.saved_model.load(nn_model_path)
        with open(nn_data_path, 'rb') as f:
            loaded_data = pickle.load(f)
            data['neural_network'] = {
                'feature_names': loaded_data.get('feature_names', []),
                'genre_encoder': loaded_data.get('genre_encoder'),
                'type_encoder': loaded_data.get('type_encoder'),
                'source_encoder': loaded_data.get('source_encoder'),
                'demographic_encoder': loaded_data.get('demographic_encoder'),
                'scaler': loaded_data.get('scaler'),
                'original_data': loaded_data
            }
            if 'X' in loaded_data and isinstance(loaded_data['X'], pd.DataFrame):
                total_anime = max(total_anime, loaded_data['X'].shape[0])
    except Exception as e:
        st.warning(f"Failed to load neural network model or data: {str(e)}")
    
    if total_anime == 0:
        total_anime = 5000

    return models, data, total_anime