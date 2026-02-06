import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_engineer_features(path):
    """
    Creates logical inconsistency features and cleans data for Anomaly Detection.
    """
    df = pd.read_csv(path)
    
    # 1. Financial Stress Ratios
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-5)
    
    # 2. Logic Inconsistency: External Sources Conflict
    # Normal people have consistent scores. Anomalies have high variance (STD).
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCES_STD'] = df[ext_sources].std(axis=1)
    df['EXT_SOURCES_MEAN'] = df[ext_sources].mean(axis=1)
    
    # 3. Employment & Age Consistency
    # Treat 365243 as 0 stability
    df['DAYS_EMPLOYED_CLEAN'] = df['DAYS_EMPLOYED'].replace(365243, 0)
    df['EMPLOYED_AGE_RATIO'] = abs(df['DAYS_EMPLOYED_CLEAN'] / (df['DAYS_BIRTH'] + 1e-5))
    
    # 4. Selection of meaningful features (Strictly No Leakage)
    selected_features = [
        'EXT_SOURCES_MEAN', 'EXT_SOURCES_STD',
        'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
        'EMPLOYED_AGE_RATIO', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH'
    ]
    
    meta_cols = ['SK_ID_CURR', 'TARGET']
    
    # Fill missing values with median
    data = df[selected_features].copy()
    for col in data.columns:
        data[col] = data[col].fillna(data[col].median())
        
    return data, df[meta_cols]

def preprocess_data(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler