import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
import os
import logging
from typing import Tuple, Optional

# Import functions from the data processor module
from data_processor import create_dummy_data, preprocess_data, train_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
PROCESSED_DATA_PATH = 'nhl_processed_injury_data.pkl'
MODEL_PATH = 'nhl_injury_model.joblib'
SCALER_PATH = 'nhl_injury_scaler.joblib'
FEATURE_NAMES_PATH = 'feature_names.joblib'

# Define the feature names explicitly
FEATURE_NAMES = ['Age', 'GP', 'G', 'A', 'P', '+/-', 'PIM', 'TOI', 'Position_Forward', 'Position_Defenseman', 'Position_Goalie', 'SV%', 'GAA']

@st.cache_data
def load_or_create_data() -> Optional[pd.DataFrame]:
    if os.path.exists(PROCESSED_DATA_PATH):
        try:
            df = pd.read_pickle(PROCESSED_DATA_PATH)
            return clean_data(df)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    else:
        st.info("Processed data not found. Creating new dummy data.")
        df = create_dummy_data()
        X, y, df_processed = preprocess_data(df)
        df_processed.to_pickle(PROCESSED_DATA_PATH)
        return clean_data(df_processed)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by handling NaN values."""
    # Fill NaN values with mean for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    # For categorical columns (if any), fill with mode
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])
    
    return df

@st.cache_resource
def load_or_train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[RandomForestClassifier], Optional[StandardScaler]]:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURE_NAMES_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            saved_feature_names = joblib.load(FEATURE_NAMES_PATH)
            if not all(X.columns == saved_feature_names):
                raise ValueError("Loaded feature names do not match current data")
            return model, scaler
        except Exception as e:
            st.error(f"Error loading model, scaler, or feature names: {e}")
    
    st.info("Model not found or feature mismatch. Training new model.")
    X = X[FEATURE_NAMES]  # Ensure consistent feature order
    model, scaler = train_model(X, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(X.columns, FEATURE_NAMES_PATH)
    return model, scaler

def predict_injury_risk(model: RandomForestClassifier, scaler: StandardScaler, df_processed: pd.DataFrame, input_data: dict) -> Tuple[Optional[int], Optional[float]]:
    if model is None or scaler is None:
        st.error("Model or scaler not available. Unable to make prediction.")
        return None, None

    new_data = pd.DataFrame([input_data])
    new_data = new_data.reindex(columns=FEATURE_NAMES, fill_value=0)
    
    X_new_scaled = scaler.transform(new_data)
    risk_probability = model.predict_proba(X_new_scaled)[0][1]
    risk_prediction = model.predict(X_new_scaled)[0]
    return risk_prediction, risk_probability

def predict_injury_risk_for_all(model: RandomForestClassifier, scaler: StandardScaler, df_processed: pd.DataFrame) -> Optional[pd.DataFrame]:
    if model is None or scaler is None:
        st.error("Model or scaler not available. Unable to make predictions.")
        return None

    X = df_processed[FEATURE_NAMES]
    X_scaled = scaler.transform(X)
    y_probs = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_probs > 0.5).astype(int)
    return pd.DataFrame({
        'player_name': df_processed['player_name'],
        'Injury_Risk_Prediction': y_pred,
        'Injury_Risk_Probability': y_probs
    })

def plot_feature_importance(model: RandomForestClassifier, feature_names: np.ndarray, top_n: int = 15):
    if model is None:
        st.error("Model not available. Unable to plot feature importance.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = feature_names[indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, ax=ax)
    ax.set_title(f'Top {top_n} Feature Importances')
    st.pyplot(fig)  
def display_player_stats(player_data: pd.DataFrame):
    st.subheader("Player Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Age:** {player_data['Age'].iloc[0] if 'Age' in player_data.columns else 'N/A'}")
        st.markdown(f"**Games Played:** {player_data['GP'].iloc[0] if 'GP' in player_data.columns else 'N/A'}")
        
        # Determine position based on available statistics
        if 'Position_Forward' in player_data.columns and player_data['Position_Forward'].iloc[0] == 1:
            position = 'Forward'
        elif 'Position_Defenseman' in player_data.columns and player_data['Position_Defenseman'].iloc[0] == 1:
            position = 'Defenseman'
        elif 'Position_Goalie' in player_data.columns and player_data['Position_Goalie'].iloc[0] == 1:
            position = 'Goalie'
        else:
            position = 'Unknown'
        st.markdown(f"**Position:** {position}")
    
    with col2:
        if position == 'Goalie':
            st.markdown("**Goalie Performance:**")
            st.metric("Save %", f"{player_data['SV%'].iloc[0]:.3f}" if 'SV%' in player_data.columns else 'N/A')
            st.metric("GAA", f"{player_data['GAA'].iloc[0]:.2f}" if 'GAA' in player_data.columns else 'N/A')
        else:
            st.markdown("**Skater Performance:**")
            st.metric("Points", f"{player_data['P'].iloc[0]}" if 'P' in player_data.columns else 'N/A')
            st.metric("+/-", f"{player_data['+/-'].iloc[0]}" if '+/-' in player_data.columns else 'N/A')
            st.metric("TOI/G", f"{player_data['TOI'].iloc[0]:.2f}" if 'TOI' in player_data.columns else 'N/A')

def get_injury_prevention_recommendation(injury_risk_pred: int) -> str:
    if injury_risk_pred == 1:
        return ("Injury Prevention Recommendation: "
                "Implement a tailored injury prevention program focusing on high-risk areas. "
                "This may include specialized strength and conditioning exercises, "
                "on-ice biomechanical analysis to identify and correct potential issues, "
                "and a carefully managed ice time and practice load to prevent overuse injuries.")
    else:
        return ("Injury Prevention Recommendation: "
                "Continue with the current training regimen while monitoring for any signs of fatigue or discomfort. "
                "Regular check-ins with the medical staff, proper nutrition, and adequate rest "
                "will help maintain the player's low injury risk status.")

def main():
    st.set_page_config(page_title="NHL Injury Risk Prediction App", page_icon="🏒")
    st.title("NHL Injury Risk Prediction App 🏒")
    
    df_processed = load_or_create_data()
    
    if df_processed is None:
        st.warning("Data not available. Some features may be limited.")
        return
    
    X = df_processed[FEATURE_NAMES]
    y = df_processed['Injured']
    model, scaler = load_or_train_model(X, y)
    
    if model is None or scaler is None:
        st.warning("Model or scaler not available. Predictions cannot be made.")
        return
    
    menu = ["Home", "Predict Injury Risk", "Player Risk Lookup", "Data Visualization", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Welcome to the NHL Injury Risk Prediction App")
        st.write("""
        This app uses machine learning to predict the injury risk for NHL players based on their statistics. 
        Use the sidebar to navigate through different features of the app:
        
        - **Predict Injury Risk**: Input player stats to get an injury risk prediction
        - **Player Risk Lookup**: Select a player to see their injury risk, stats, and prevention recommendations
        - **Data Visualization**: Explore feature importance and SHAP values
        - **About**: Learn more about how this app works
        
        Get started by selecting an option from the sidebar!
        """)
    
    elif choice == "Predict Injury Risk":
        st.subheader("Predict Injury Risk for a Player")
        
        with st.form("player_form"):
            player_name = st.text_input("Player Name")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=45, value=25)
                games = st.number_input("Games Played", min_value=1, max_value=82, value=60)
                position = st.selectbox("Position", ["Forward", "Defenseman", "Goalie"])
            with col2:
                if position in ["Forward", "Defenseman"]:
                    points = st.number_input("Points", min_value=0, value=30)
                    plus_minus = st.number_input("Plus/Minus", min_value=-82, max_value=82, value=0)
                    toi = st.number_input("Average Time on Ice (minutes)", min_value=0.0, max_value=30.0, value=15.0)
                else:  # Goalie
                    save_percentage = st.number_input("Save Percentage", min_value=0.800, max_value=1.000, value=0.910, format="%.3f")
                    gaa = st.number_input("Goals Against Average", min_value=0.00, max_value=5.00, value=2.50)
            
            submitted = st.form_submit_button("Predict")
        
        if submitted:
            input_data = {
                'player_name': player_name,
                'Age': age,
                'GP': games,
                'Position_Forward': 1 if position == "Forward" else 0,
                'Position_Defenseman': 1 if position == "Defenseman" else 0,
                'Position_Goalie': 1 if position == "Goalie" else 0,
                'P': points if position in ["Forward", "Defenseman"] else 0,
                '+/-': plus_minus if position in ["Forward", "Defenseman"] else 0,
                'TOI': toi if position in ["Forward", "Defenseman"] else 0,
                'SV%': save_percentage if position == "Goalie" else 0,
                'GAA': gaa if position == "Goalie" else 0
            }
            
            risk_prediction, risk_probability = predict_injury_risk(model, scaler, df_processed, input_data)
            if risk_prediction is not None and risk_probability is not None:
                risk_label = 'High Risk' if risk_prediction == 1 else 'Low Risk'
                st.markdown(f"### Injury Risk Prediction for {player_name}: {risk_label}")
                st.markdown(f"#### Risk Probability: {risk_probability:.2f}")
                
                recommendation = get_injury_prevention_recommendation(risk_prediction)
                st.markdown("---")
                st.markdown(f"### {recommendation}")
            else:
                st.error("Unable to make prediction. Please check the input data and try again.")
    
    elif choice == "Player Risk Lookup":
        st.subheader("NHL Players Injury Risk Lookup")
        df_results = predict_injury_risk_for_all(model, scaler, df_processed)
        if df_results is not None:
            player_names = df_processed['player_name'].unique()
            selected_player = st.selectbox("Select a Player", sorted(player_names))
            player_data = df_processed[df_processed['player_name'] == selected_player]
            if not player_data.empty:
                player_result = df_results[df_results['player_name'] == selected_player].iloc[0]
                injury_risk_prob = player_result['Injury_Risk_Probability']
                injury_risk_pred = player_result['Injury_Risk_Prediction']
                risk_label = 'High Risk' if injury_risk_pred == 1 else 'Low Risk'
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"<h2 style='font-size: 24px;'>{selected_player}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='font-size: 20px;'>Injury Risk: <strong>{risk_label}</strong></h3>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='font-size: 20px;'>Risk Probability: <strong>{injury_risk_prob:.2f}</strong></h3>", unsafe_allow_html=True)
                
                with col2:
                    display_player_stats(player_data)
                
                recommendation = get_injury_prevention_recommendation(injury_risk_pred)
                st.markdown("---")
                st.markdown(f"### {recommendation}")
            else:
                st.write("Player data not found.")
        else:
            st.error("Unable to generate risk predictions for players.")

    elif choice == "Data Visualization":
        st.subheader("Data Visualization")
        
        st.write("""
        This section provides insights into how our model makes predictions. We use feature importance visualization.
        """)
        
        st.subheader("Feature Importance")
        st.write("""
        Feature importance shows how much each feature contributes to the model's predictions. 
        Features with higher importance have a greater impact on the injury risk prediction.
        """)
        plot_feature_importance(model, X.columns, top_n=15)
        
        st.subheader("Additional Feature Analysis")
        st.write("""
        Let's take a closer look at how individual features relate to injury risk.
        """)
        
        # Select a feature for analysis
        feature_to_analyze = st.selectbox("Select a feature to analyze:", FEATURE_NAMES)
        
        # Create a plot showing the relationship between the selected feature and injury risk
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Injured', y=feature_to_analyze, data=df_processed, ax=ax)
        ax.set_title(f'Relationship between {feature_to_analyze} and Injury Risk')
        ax.set_xlabel('Injured (0 = No, 1 = Yes)')
        ax.set_ylabel(feature_to_analyze)
        st.pyplot(fig)
        
        st.write(f"""
        This box plot shows the distribution of {feature_to_analyze} for players who were not injured (0) and those who were injured (1).
        The box represents the interquartile range (IQR), the line inside the box is the median, and the whiskers extend to show the rest of the distribution.
        Points beyond the whiskers are outliers.
        
        Interpreting the plot:
        - If the boxes for 0 and 1 are at different levels, it suggests that {feature_to_analyze} might be associated with injury risk.
        - If the boxes overlap significantly, it might indicate that {feature_to_analyze} alone is not strongly predictive of injury risk.
        - Keep in mind that our model considers multiple features together, which can reveal patterns that aren't visible when looking at single features.
        """)
        
        st.write("""
        Note: We've replaced the SHAP values visualization due to compatibility issues with the current version of NumPy. 
        SHAP values provide a more detailed view of feature importance, showing how each feature impacts the model output for each prediction.
        If you're interested in SHAP values, you might need to use a different version of the libraries or explore alternative explanation methods.
        """)
        explainer = shap.TreeExplainer(model)
        
        # Prepare the data for SHAP
        X_sample = X.sample(min(100, len(X)))
        X_sample_scaled = scaler.transform(X_sample)
        
        # Generate SHAP values
        shap_values = explainer.shap_values(X_sample_scaled)
        
        # Ensure shap_values is a list with at least two elements
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_to_plot, X_sample, feature_names=X_sample.columns, plot_type="bar", show=False)
        st.pyplot(fig)
    
    elif choice == "About":
        st.subheader("About the NHL Injury Risk Prediction App")
        st.write("""
        This app predicts the injury risk of NHL players based on historical data and player statistics. It uses a Random Forest Classifier 
        trained on player performance data and simulated injury data.

        Key features of the app include:
        1. Injury risk prediction for individual players
        2. Player risk lookup for all players in the database
        3. Data visualizations to understand feature importance and model predictions
        4. Injury prevention recommendations based on predicted risk

        How it works:
        - The app uses a machine learning model (Random Forest Classifier) trained on historical player data.
        - It considers various factors such as age, games played, position, and performance statistics to assess injury risk.
        - The model outputs a probability of injury risk, which we then classify as either 'High Risk' or 'Low Risk'.
        - Based on this risk assessment, the app provides tailored injury prevention recommendations.

        Limitations and Considerations:
        - This model is based on simulated injury data. In a real-world scenario, you would need actual historical injury data for more accurate predictions.
        - The predictions should be used as one of many tools in assessing player health and should not replace medical expertise or individual player assessments.
        - Factors not captured in the data (e.g., recent injuries, off-ice training, etc.) may affect actual injury risk.
        - Always consult with medical professionals and team experts for comprehensive player evaluations and injury prevention strategies.

        Data sources:
        - Player statistics: Simulated data based on typical NHL player statistics
        - Injury data: Simulated for demonstration purposes

        Future Improvements:
        - Incorporate real NHL data and actual injury records for more accurate predictions
        - Include more advanced statistics and metrics (e.g., advanced analytics, biometric data)
        - Develop position-specific models for more tailored predictions
        - Implement time-series analysis to account for changes in player performance and risk over time

        For more information on the data processing and model training, please refer to the `data_processor.py` script.

        Disclaimer: This app is for educational and demonstration purposes only. It should not be used as a sole basis for medical decisions or player management in real-world scenarios.
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.exception("An error occurred in the Streamlit app")
