import streamlit as st
import numpy as np
import pandas as pd
import pickle
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import itertools
from utils import load_models_and_data

models, data, total_anime = load_models_and_data()

# with open('./models/overall_best_combination.pkl', 'rb') as f:
#     overall_best_combo, overall_best_rank = pickle.load(f)

# with open('./models/specific_best_combinations.pkl', 'rb') as f:
#     specific_best_combos = pickle.load(f)

st.title("Anime Popularity Predictor")

st.markdown("""<style>.stTabs [data-baseweb="tab-list"] {justify-content: flex-end;}</style>""", unsafe_allow_html=True)

genres = ["Action", "Adventure", "Comedy", "Drama", "Sci-Fi", "Fantasy", "Romance", "Slice of Life",
          "Mystery", "Supernatural", "Sports", "Historical", "Horror", "Psychological", "Thriller",
          "Ecchi", "Mecha", "Music", "Harem", "Gourmet", "Parody", "Dementia", "Super Power", "School",
          "Josei", "Vampire", "Hentai", "Police", "Space", "Demons", "Martial Arts", "Military", "Cars",
          "Samurai", "Magic", "Kids", "Game", "Shoujo Ai", "Shounen Ai", "Yaoi", "Yuri", "Isekai",
          "Seinen", "Shounen"]

types = ["TV", "Movie", "OVA", "ONA", "Special", "Music", "PV", "CM", "TV Special"]

sources = ["Original", "Manga", "Light novel", "Visual novel", "Game", "Novel", "4-koma manga", "Book",
           "Card game", "Music", "Mixed media", "Picture book", "Web manga", "Other"]

demographics = ["Shounen", "Seinen", "Shoujo", "Josei", "Kids"]

all_features = {
    'type': types,
    'source': sources,
    'demographic': demographics,
    'genre': genres,
    'producers_count': list(range(1, 21)),
    'platform_count': list(range(1, 11))
}

def preprocess_features(features, model_key, model, model_data, total_anime):
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
    processed_features = preprocess_features(features, model_key, model, model_data, total_anime)
    
    prediction = model.predict(processed_features)[0]

    try:
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(processed_features)
            confidence_interval = np.percentile(y_pred_proba, [2.5, 97.5], axis=1).flatten()
            ci_lower, ci_upper = confidence_interval
        else:
            std_dev = np.std([model.predict(processed_features) for _ in range(100)])
            ci_lower = max(1, prediction - 1.96 * std_dev)
            ci_upper = min(total_anime, prediction + 1.96 * std_dev)
    except Exception as e:
        print(f"Error calculating confidence interval: {str(e)}")
        ci_lower = max(1, prediction * 0.9)
        ci_upper = min(total_anime, prediction * 1.1)
    
    return prediction, ci_lower, ci_upper

def monte_carlo_simulation(model_key, features, num_simulations=1000):
    model = models[model_key]
    model_data = data[model_key]
    results = []
    for _ in range(num_simulations):
        sim_features = features.copy()
        sim_features['producers_count'] = np.random.randint(1, 21)
        sim_features['platform_count'] = np.random.randint(1, 11)
        
        processed_features = preprocess_features(sim_features, model_key, model, model_data, total_anime)
        prediction = model.predict(processed_features)[0]
        results.append(prediction)
    
    return np.array(results)

tab1, tab2 = st.tabs(["Prediction", "Optimal Combinations"])

with tab1:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Categorical Features")
        genre = st.multiselect("Genre", options=genres)
        anime_type = st.selectbox("Type", options=types)

    with col2:
        st.subheader(" ")
        source = st.selectbox("Source", options=sources)
        demographic = st.selectbox("Demographic", options=demographics)

    col3, col4 = st.columns(2)

    with col3:
        model_options = {
            'multiple_regression_type_source_demographic_producer_genre_platform': 'Multiple Linear Regression',
            'random_forest': 'Random Forest',
            'gradient_boosting': 'Gradient Boosting',
            'xgboost': 'XGBoost'
        }
        selected_model = st.selectbox("Select Model", options=list(model_options.values()))

    with col4:
        st.write('')
        st.write('')
        predict_button = st.button("Predict Popularity")

    if predict_button:
        selected_model_key = [k for k, v in model_options.items() if v == selected_model][0]
        st.session_state.selected_model_key = selected_model_key
        
        features = pd.DataFrame({
            'type': [anime_type],
            'source': [source],
            'demographic': [demographic],
            'genre': [genre],
            'producers_count': [1],
            'platform_count': [1]
        })
        
        mc_results = monte_carlo_simulation(selected_model_key, features)
        prediction = np.mean(mc_results)
        ci_lower, ci_upper = np.percentile(mc_results, [2.5, 97.5])
        
        st.header("Prediction Results")
        percentile = (prediction - 1) / (total_anime - 1) * 100
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Rank", f"{prediction:.0f}")
        with col2:
            st.metric("Estimated Range", f"{ci_lower:.0f} - {ci_upper:.0f}")
        with col3:
            st.metric("Percentile", f"Top {percentile:.1f}%")

        fig = go.Figure()
        
        fig.add_trace(go.Indicator(
            mode="number+gauge+delta",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': total_anime*0.5, 'position': "top", 'relative': True},
            title={'text': "Predicted Popularity Rank"},
            gauge={
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
            x=[0, 1],
            y=[ci_lower, ci_lower],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Lower Estimate'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[ci_upper, ci_upper],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Upper Estimate'
        ))

        fig.update_layout(
            height=700,
            yaxis=dict(
                range=[total_anime, 1],
                autorange="reversed",
                zeroline=False,
                title="Anime Rank" 
            ),
            showlegend=False
        )

        st.plotly_chart(fig)

        st.subheader("Monte Carlo Simulation")
        mc_prediction = np.mean(mc_results)
        mc_ci_lower, mc_ci_upper = np.percentile(mc_results, [2.5, 97.5])
        
        st.write(f"Monte Carlo Prediction: {mc_prediction:.0f}")
        st.write(f"Monte Carlo 95% Confidence Interval: {mc_ci_lower:.0f} - {mc_ci_upper:.0f}")
        
        fig_mc = px.histogram(mc_results, nbins=50, title="Monte Carlo Simulation Results")
        fig_mc.add_vline(x=mc_prediction, line_dash="dash", line_color="red", annotation_text="Mean Prediction")
        st.plotly_chart(fig_mc)

with tab2:
    st.header("Optimal Feature Combinations")

    st.subheader("1. Optimize for a Specific Feature")
    col1, col2 = st.columns(2)
    with col1:
        fixed_feature = st.selectbox("Choose a feature to fix", ['type', 'source', 'demographic', 'genre'])

    with col2:
        fixed_value = st.selectbox(f"Choose {fixed_feature}", all_features[fixed_feature])

    # if st.button("Show Optimal Combination"):
    #     best_combo, best_rank = specific_best_combos[fixed_feature][fixed_value]
    #     st.write(f"Best possible rank with {fixed_feature} = {fixed_value}: {best_rank:.0f}")
    #     st.write("Optimal combination:")
    #     for feature, value in best_combo.items():
    #         if feature != fixed_feature:
    #             st.write(f"- {feature}: {value}")

    #         fig = go.Figure(data=[go.Bar(
    #             x=list(best_combo.keys()),
    #             y=[1 if isinstance(v, str) else v/max(all_features[k]) for k, v in best_combo.items()],
    #             text=list(best_combo.values()),
    #             textposition='auto',
    #         )])
    #         fig.update_layout(title=f"Optimal Combination for {fixed_feature} = {fixed_value}", 
    #                           xaxis_title="Features", 
    #                           yaxis_title="Normalized Value")
    #         st.plotly_chart(fig)

    #     st.subheader("2. Overall Optimal Combination")
    #     if st.button("Show Overall Optimal Combination"):
    #         st.write(f"Best possible rank: {overall_best_rank:.0f}")
    #         st.write("Optimal combination:")
    #         for feature, value in overall_best_combo.items():
    #             st.write(f"- {feature}: {value}")

    #     fig = go.Figure(data=[go.Bar(
    #         x=list(best_combo.keys()),
    #         y=[1 if isinstance(v, str) else v / max(all_features[k]) for k, v in best_combo.items()],
    #         text=list(best_combo.values()),
    #         textposition='auto',
    #         marker=dict(
    #             line=dict(color='darkblue', width=2)
    #         )
    #     )])
    #     fig.update_layout(title="Overall Optimal Combination", 
    #                         xaxis_title="Features", 
    #                         yaxis_title="Normalized Value")
    #     st.plotly_chart(fig)