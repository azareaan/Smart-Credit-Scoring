import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.fuzzy_logic import create_fuzzy_system
import os

def main():
    INPUT_FILE = 'outputs/fuzzy_ready_data.csv'
    OUTPUT_FILE = 'outputs/final_risk_scored.csv'
    
    print("--- STEP 1: Loading Data for Fuzzy Inference ---")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Please run main.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} customers.")

    # Prepare inputs
    # Ensure inputs are clipped between 0 and 1 just in case
    anomalies = df['Anomaly_Score_Fuzzy'].clip(0, 1).values
    # Handle missing values in EXT_SOURCES_MEAN by filling with 0.5 (Neutral)
    ext_sources = df['EXT_SOURCES_MEAN'].fillna(0.5).clip(0, 1).values
    
    print("--- STEP 2: Running Fuzzy Logic System (This may take a moment) ---")
    fuzzy_system = create_fuzzy_system()
    
    risk_scores = []
    
    # Using tqdm for a progress bar
    for i in tqdm(range(len(df)), desc="Calculating Risks"):
        try:
            # Pass inputs to the fuzzy system
            fuzzy_system.input['anomaly_score'] = anomalies[i]
            fuzzy_system.input['ext_source'] = ext_sources[i]
            
            # Compute result
            fuzzy_system.compute()
            risk_scores.append(fuzzy_system.output['risk'])
        except:
            # Fallback for edge cases
            risk_scores.append(50)

    df['FINAL_RISK_SCORE'] = risk_scores
    
    print("--- STEP 3: Analysis & Visualization ---")
    
    # Define Risk Categories
    df['RISK_CATEGORY'] = pd.cut(df['FINAL_RISK_SCORE'], 
                                 bins=[0, 40, 70, 100], 
                                 labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    print("\n📊 Risk Category Distribution:")
    print(df['RISK_CATEGORY'].value_counts())
    
    # Save Plot
    plt.figure(figsize=(10, 6))
    plt.hist(df[df['TARGET']==0]['FINAL_RISK_SCORE'], bins=50, alpha=0.5, label='Repayers (Target=0)', color='green')
    plt.hist(df[df['TARGET']==1]['FINAL_RISK_SCORE'], bins=50, alpha=0.5, label='Defaulters (Target=1)', color='red')
    plt.title('Final Fuzzy Risk Score Distribution')
    plt.xlabel('Risk Score (0-100)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('outputs/figures/final_risk_distribution.png')
    print("💾 Risk Plot saved to outputs/figures/final_risk_distribution.png")
    
    # Save Final CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"🚀 Project Complete! Final Report: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()