import os
import pandas as pd
import numpy as np
from src.features import load_and_engineer_features, preprocess_data
from src.model import Autoencoder
from src.train import train_autoencoder, get_reconstruction_error
from sklearn.metrics import roc_auc_score

def main():
    # Configuration
    DATA_PATH = 'data/application_train.csv'
    OUTPUT_DIR = 'outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("--- STEP 1: Engineering Features (Logical Consistency) ---")
    X, meta = load_and_engineer_features(DATA_PATH)
    
    # Semi-Supervised Logic: Train ONLY on Target 0
    X_normal = X[meta['TARGET'] == 0]
    X_normal_scaled, scaler = preprocess_data(X_normal)
    X_full_scaled = scaler.transform(X)
    
    print(f"Dataset ready. Training AE on {len(X_normal)} repayers...")

    print("--- STEP 2: Training Autoencoder ---")
    model = Autoencoder(input_dim=X_normal_scaled.shape[1])
    trained_model = train_autoencoder(model, X_normal_scaled, epochs=40)

    print("--- STEP 3: Evaluating & Exporting for Fuzzy ---")
    mse_scores = get_reconstruction_error(trained_model, X_full_scaled)
    
    # Normalize MSE to [0, 1] for Fuzzy Membership Degrees
    fuzzy_score = (mse_scores - mse_scores.min()) / (mse_scores.max() - mse_scores.min() + 1e-8)
    
    results_df = meta.copy()
    results_df['Anomaly_Score_Fuzzy'] = fuzzy_score
    
    # Real World Evaluation (Correlation with Default)
    auc = roc_auc_score(results_df['TARGET'], results_df['Anomaly_Score_Fuzzy'])
    print(f"\n✅ Clean Model AUC: {auc:.4f}")

    # Final Export for Fuzzy Logic System
    final_output = pd.concat([results_df, X], axis=1)
    final_output.to_csv(f'{OUTPUT_DIR}/fuzzy_ready_data.csv', index=False)
    print(f"🚀 Success! Final data saved in {OUTPUT_DIR}/fuzzy_ready_data.csv")

if __name__ == "__main__":
    main()