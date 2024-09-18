import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_PATH = 'nhl_data.pkl'
MODEL_PATH = 'nhl_injury_model.joblib'
SCALER_PATH = 'nhl_injury_scaler.joblib'
PROCESSED_DATA_PATH = 'nhl_processed_injury_data.pkl'

def load_and_process_data():
    logging.info("Starting data loading and processing")
    
    # TODO: Replace this with actual NHL data loading
    # For now, we'll create a dummy dataset
    data = {
        'player_name': [f'Player_{i}' for i in range(100)],
        'Age': np.random.randint(18, 40, 100),
        'GP': np.random.randint(1, 82, 100),
        'G': np.random.randint(0, 50, 100),
        'A': np.random.randint(0, 70, 100),
        'P': np.random.randint(0, 100, 100),
        '+/-': np.random.randint(-30, 30, 100),
        'PIM': np.random.randint(0, 150, 100),
        'TOI': np.random.uniform(5, 25, 100),
        'Position': np.random.choice(['Forward', 'Defenseman', 'Goalie'], 100)
    }
    df = pd.DataFrame(data)
    
    # Add some goalie-specific stats
    mask = df['Position'] == 'Goalie'
    df.loc[mask, 'SV%'] = np.random.uniform(0.880, 0.930, mask.sum())
    df.loc[mask, 'GAA'] = np.random.uniform(2.0, 3.5, mask.sum())
    
    logging.info(f"Created dummy dataset with {len(df)} players")
    
    # Simulate injury data
    df['Injured'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])  # 10% injury rate
    
    logging.info("Data loading and initial processing completed")
    return df

def preprocess_data(df):
    logging.info("Starting data preprocessing")
    
    # Create dummy variables for Position
    df = pd.get_dummies(df, columns=['Position'], prefix='Position')
    
    # List of features to use
    features = ['Age', 'GP', 'G', 'A', 'P', '+/-', 'PIM', 'TOI', 
                'Position_Forward', 'Position_Defenseman', 'Position_Goalie', 
                'SV%', 'GAA']
    
    # Select only available features
    available_features = [col for col in features if col in df.columns]
    
    X = df[available_features]
    y = df['Injured']
    
    # Check for NaN values
    nan_columns = X.columns[X.isna().any()].tolist()
    if nan_columns:
        logging.warning(f"NaN values found in columns: {nan_columns}")
        
        # Impute NaN values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        logging.info("NaN values have been imputed with mean values")
    
    logging.info(f"Preprocessed data shape: {X.shape}")
    logging.info(f"Features used: {available_features}")
    
    return X, y, df

def train_model(X, y):
    logging.info("Starting model training")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    X_test_scaled = scaler.transform(X_test)
    accuracy = model.score(X_test_scaled, y_test)
    logging.info(f"Model accuracy on test set: {accuracy:.2f}")
    
    logging.info("Model training completed")
    return model, scaler

def main():
    # Load and process data
    df = load_and_process_data()
    df.to_pickle(DATA_PATH)
    logging.info(f"Raw data saved to {DATA_PATH}")
    
    # Preprocess data
    X, y, df_processed = preprocess_data(df)
    df_processed.to_pickle(PROCESSED_DATA_PATH)
    logging.info(f"Processed data saved to {PROCESSED_DATA_PATH}")
    
    # Train model
    model, scaler = train_model(X, y)
    
    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")
    logging.info(f"Scaler saved to {SCALER_PATH}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("An error occurred in the data processing script")
        raise