import streamlit as st
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

st.title("Anime Popularity Predictor")

@st.cache_resource
def load_models_and_data():
    models = {}
    data = {}
    
    model_names = ['multiple_regression_type_source_demographic_producer_genre_platform', 'random_forest', 'gradient_boosting', 'xgboost']
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

models, data = load_models_and_data()

model_display_names = {
    'multiple_regression_type_source_demographic_producer_genre_platform': 'Multiple Linear Regression',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
    'xgboost': 'XGBoost',
    'neural_network': 'Neural Network'
}

st.header("Enter Anime Details")

producers_count = st.number_input("Number of Producers", min_value=1, max_value=20, value=1)
anime_type = st.selectbox("Type", options=["TV", "Movie", "OVA", "ONA", "Special"])
source = st.selectbox("Source", options=["Original", "Manga", "Light novel", "Visual novel", "Game", "Other"])
demographic = st.selectbox("Demographic", options=["Shounen", "Seinen", "Shoujo", "Josei", "Kids"])
genre = st.multiselect("Genre", options=["Action", "Adventure", "Comedy", "Drama", "Sci-Fi", "Fantasy", "Romance", "Slice of Life"])
platform_count = st.number_input("Number of Platforms", min_value=1, max_value=10, value=1)

st.header("Choose Prediction Model")
selected_model = st.selectbox("Select Model", options=list(model_display_names.values()))
st.write("Recommended model: XGBoost")

selected_model_key = [k for k, v in model_display_names.items() if v == selected_model][0]

def predict_popularity(model_key, features):
    model = models[model_key]
    if model_key == 'neural_network':
        return model.predict(features.values)
    else:
        return model.predict(features)

def get_confidence_interval(model_key, features, confidence=0.95):
    prediction = predict_popularity(model_key, features)
    if model_key == 'multiple_regression_type_source_demographic_producer_genre_platform':
        _, std_dev = models[model_key].predict(features, return_std=True)
    else:
        std_dev = np.std([predict_popularity(model_key, features) for _ in range(100)])
    
    confidence_interval = stats.t.interval(confidence, len(features)-1, loc=prediction, scale=std_dev)
    return confidence_interval

def monte_carlo_simulation(model_key, features, n_simulations=1000):
    results = []
    for _ in range(n_simulations):
        simulated_features = features.copy()
        simulated_features['producers_count'] += np.random.randint(0, 10)
        simulated_features['platform_count'] += np.random.randint(0, 12)
        
        simulated_features['producers_count'] = max(1, simulated_features['producers_count'])
        simulated_features['platform_count'] = max(1, simulated_features['platform_count'])
        
        prediction = predict_popularity(model_key, simulated_features)
        results.append(prediction)
    
    return np.mean(results), np.std(results)

if st.button("Predict Popularity"):
    features = pd.DataFrame({
        'producers_count': [producers_count],
        'type': [anime_type],
        'source': [source],
        'demographic': [demographic],
        'genre': [','.join(genre)],
        'platform_count': [platform_count]
    })
    
    scaler = data[selected_model_key].get('scaler')
    if scaler:
        features = pd.DataFrame(scaler.transform(features), columns=features.columns)

    prediction = predict_popularity(selected_model_key, features)
    ci = get_confidence_interval(selected_model_key, features)
    mc_mean, mc_std = monte_carlo_simulation(selected_model_key, features)
    
    st.header("Prediction Results")
    st.write(f"Predicted Popularity Rank: {prediction[0]:.0f}")
    st.write(f"95% Confidence Interval: {ci[0][0]:.0f} - {ci[0][1]:.0f}")
    st.write(f"Monte Carlo Simulation Results:")
    st.write(f"Mean: {mc_mean:.0f}")
    st.write(f"Standard Deviation: {mc_std:.0f}")