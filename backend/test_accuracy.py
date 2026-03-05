import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def test_model_accuracy():
    # Define paths relative to the location of this script (test_accuracy.py)
    # Since test_accuracy.py is in 'backend', and 'models' is inside 'backend'
    models_dir = 'models' # Changed from 'backend/models'
    data_dir = 'data'     # Changed from 'backend/data'
    
    parkinsons_hospital_path = os.path.join(data_dir, 'parkinsons_hospital.csv')

    try:
        model = joblib.load(os.path.join(models_dir, 'parkinsons_predictor.pkl'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        feature_names = joblib.load(os.path.join(models_dir, 'feature_names.pkl'))
        print("Model, scaler, and feature names loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading model components: {e}")
        print("Please ensure 'parkinsons_predictor.pkl', 'scaler.pkl', and 'feature_names.pkl' are in the 'backend/models/' directory.")
        print("Run ml_models.py first to train and save the model.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading model components: {e}")
        return

    print(f"Loading test data from: {parkinsons_hospital_path}")
    try:
        df_test = pd.read_csv(parkinsons_hospital_path)
        print("Test data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure 'parkinsons_hospital.csv' is in the 'backend/data' directory.")
        return
    
    # Drop the 'name' column if it exists, consistent with training preprocessing
    if 'name' in df_test.columns:
        df_test = df_test.drop('name', axis=1)

    # Separate features (X) and target (y)
    if 'status' not in df_test.columns:
        print("Error: 'status' column not found in the test DataFrame.")
        return
        
    X_test = df_test.drop('status', axis=1)
    y_test = df_test['status']

    # Ensure test data has the same features as the training data, in the same order
    try:
        X_test_ordered = X_test[feature_names]
    except KeyError as e:
        print(f"Error: Test data is missing a feature that the model was trained on: {e}")
        print(f"Expected features: {feature_names}")
        print(f"Features in test data: {X_test.columns.tolist()}")
        return

    # Scale the test features using the loaded scaler
    X_test_scaled = scaler.transform(X_test_ordered)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on parkinsons_hospital.csv: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    test_model_accuracy()