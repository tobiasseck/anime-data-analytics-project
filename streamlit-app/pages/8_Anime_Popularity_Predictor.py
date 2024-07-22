import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from utils import load_models_and_data

st.title("Anime Popularity Predictor")

models, data = load_models_and_data()

st.header("Anime Popularity Predictor")

producers_count = st.number_input("Number of Producers", min_value=1, max_value=20, value=1)
anime_type = st.selectbox("Type", options=["TV", "Movie", "OVA", "ONA", "Special"])
source = st.selectbox("Source", options=["Original", "Manga", "Light novel", "Visual novel", "Game", "Other"])
demographic = st.selectbox("Demographic", options=["Shounen", "Seinen", "Shoujo", "Josei", "Kids"])
genre = st.multiselect("Genre", options=["Action", "Adventure", "Comedy", "Drama", "Sci-Fi", "Fantasy", "Romance", "Slice of Life"])
platform_count = st.number_input("Number of Platforms", min_value=1, max_value=10, value=1)

model_options = {
    'multiple_regression_type_source_demographic_producer_genre_platform': 'Multiple Linear Regression',
    'random_forest': 'Random Forest',
    'gradient_boosting': 'Gradient Boosting',
    'xgboost': 'XGBoost'
}

selected_model = st.selectbox("Select Model", options=list(model_options.values()))
selected_model_key = [k for k, v in model_options.items() if v == selected_model][0]

def preprocess_features(features, model_key, model, model_data):
    expected_features = model.feature_names_in_.tolist()
    processed_features = pd.DataFrame(index=features.index)

    if 'genre' in features.columns:
        genres = features['genre'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        for char in set(''.join(expected_features)):
            if char.isalnum() or char in ["'", ' ', ',']:
                processed_features[char] = genres.str.contains(char).astype(int)

    categorical_features = ['type', 'source', 'demographic']
    for feature in categorical_features:
        if feature in features.columns:
            for expected_feature in expected_features:
                if expected_feature.startswith(f"{feature.capitalize()}_"):
                    category = expected_feature.split('_', 1)[1]
                    processed_features[expected_feature] = (features[feature] == category).astype(int)

    if 'producers_count' in features.columns:
        processed_features['Producer_Count'] = features['producers_count']
    if 'platform_count' in features.columns:
        processed_features['Platform_Count'] = features['platform_count']

    scaler = model_data.get('scaler')
    if scaler:
        numerical_features = ['Producer_Count', 'Platform_Count']
        processed_features[numerical_features] = scaler.transform(processed_features[numerical_features])

    for feature in expected_features:
        if feature not in processed_features.columns:
            processed_features[feature] = 0

    processed_features = processed_features[expected_features]

    return processed_features

def predict_popularity(model_key, features):
    model = models[model_key]
    model_data = data[model_key]
    processed_features = preprocess_features(features, model_key, model, model_data)
    
    prediction = model.predict(processed_features)[0]

    if 'original_data' in model_data and isinstance(model_data['original_data'], pd.DataFrame):
        target_data = model_data['original_data'].iloc[:, -1]
    else:
        target_data = np.arange(1, 10001)

    percentile = np.percentile(target_data, [25, 75])
    iqr = percentile[1] - percentile[0]
    
    interval_range = 0.25 * iqr
    
    ci_lower = max(1, prediction - interval_range)
    ci_upper = min(len(target_data), prediction + interval_range)
    
    return prediction, ci_lower, ci_upper, len(target_data)

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
        'genre': [genre],
        'platform_count': [platform_count]
    })
    
    prediction, ci_lower, ci_upper, total_anime = predict_popularity(selected_model_key, features)
    
    st.header("Prediction Results")
    st.write(f"Predicted Popularity Rank: {prediction:.0f}")
    st.write(f"Estimated Range: {ci_lower:.0f} - {ci_upper:.0f}")

    percentile = (prediction - 1) / (total_anime - 1) * 100
    st.write(f"This rank is in the top {percentile:.1f}% of anime.")

    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode = "number+gauge+delta",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': total_anime/2, 'position': "top"},
        title = {'text': "Predicted Popularity Rank"},
        gauge = {
            'axis': {'range': [total_anime, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [1, total_anime*0.2], 'color': 'cyan'},
                {'range': [total_anime*0.2, total_anime*0.5], 'color': 'royalblue'},
                {'range': [total_anime*0.5, total_anime], 'color': 'lightblue'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))

    fig.add_trace(go.Scatter(
        x=[0, 0.5, 1], 
        y=[total_anime, ci_lower, total_anime],
        mode='lines', 
        line=dict(color='red', dash='dash'), 
        name='Lower Estimate',
        hoverinfo='none'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 0.5, 1], 
        y=[total_anime, ci_upper, total_anime],
        mode='lines', 
        line=dict(color='red', dash='dash'), 
        name='Upper Estimate',
        hoverinfo='none'
    ))

    fig.update_layout(
        height=700,
        yaxis=dict(range=[total_anime, 1], showgrid=False, showticklabels=False, zeroline=False),
        showlegend=False
    )

    st.plotly_chart(fig)

    if hasattr(models[selected_model_key], 'feature_importances_'):
        importances = models[selected_model_key].feature_importances_
        feature_names = models[selected_model_key].feature_names_in_
        feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
        
        fig = go.Figure([go.Bar(x=feature_importance['feature'], y=feature_importance['importance'])])
        fig.update_layout(title='Top 10 Feature Importances', xaxis_title='Features', yaxis_title='Importance')
        st.plotly_chart(fig)