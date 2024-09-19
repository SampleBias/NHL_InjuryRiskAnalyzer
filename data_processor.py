import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import joblib
import logging
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DATA_PATH = 'nhl_data.pkl'
MODEL_PATH = 'nhl_injury_model.joblib'
SCALER_PATH = 'nhl_injury_scaler.joblib'
PROCESSED_DATA_PATH = 'nhl_processed_injury_data.pkl'

def create_dummy_data(n_samples: int = 100) -> pd.DataFrame:
    """Create a dummy dataset for NHL players."""
    positions = ['Forward', 'Defenseman', 'Goalie']
    data = {
        'player_name': [f'Player_{i}' for i in range(n_samples)],
        'Age': np.random.randint(18, 40, n_samples),
        'GP': np.random.randint(1, 82, n_samples),
        'G': np.random.randint(0, 50, n_samples),
        'A': np.random.randint(0, 70, n_samples),
        'P': np.random.randint(0, 100, n_samples),
        '+/-': np.random.randint(-30, 30, n_samples),
        'PIM': np.random.randint(0, 150, n_samples),
        'TOI': np.random.uniform(5, 25, n_samples),
        'Position': np.random.choice(positions, n_samples),
        'Injured': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    }
    df = pd.DataFrame(data)
    
    # Add goalie-specific stats
    mask = df['Position'] == 'Goalie'
    df.loc[mask, 'SV%'] = np.random.uniform(0.880, 0.930, mask.sum())
    df.loc[mask, 'GAA'] = np.random.uniform(2.0, 3.5, mask.sum())
    
    return df

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Preprocess the data for model training."""
    logging.info("Starting data preprocessing")
    
    # Create dummy variables for Position
    df = pd.get_dummies(df, columns=['Position'], prefix='Position')
    
    features = ['Age', 'GP', 'G', 'A', 'P', '+/-', 'PIM', 'TOI', 
                'Position_Forward', 'Position_Defenseman', 'Position_Goalie', 
                'SV%', 'GAA']
    
    # Select only available features
    available_features = [col for col in features if col in df.columns]
    
    X = df[available_features]
    y = df['Injured']
    
    # Handle NaN values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    logging.info(f"Preprocessed data shape: {X.shape}")
    logging.info(f"Features used: {available_features}")
    
    return X, y, df

def train_model(X: pd.DataFrame, y: pd.Series) -> Tuple[RandomForestClassifier, StandardScaler]:
    """Train the Random Forest model and prepare the scaler."""
    logging.info("Starting model training")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    X_test_scaled = scaler.transform(X_test)
    accuracy = model.score(X_test_scaled, y_test)
    logging.info(f"Model accuracy on test set: {accuracy:.2f}")
    
    return model, scaler

def main():
    try:
        # Create and save dummy data
        df = create_dummy_data()
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
    
    except Exception as e:
        logging.exception("An error occurred in the data processing script")
        raise

if __name__ == "__main__":
    main()