"""
Financial Data Preprocessing Pipeline
Implements the "Missing as Signal" strategy with mean imputation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class FinancialPreprocessor:
    """
    Preprocessor for Home Credit Default Risk data.
    
    Key Strategy:
    1. Missing values = informative signals (binary flags)
    2. Mean imputation (preserves distribution for StandardScaler)
    3. Feature engineering (ratios)
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = None
        self.missing_features = [
            'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'AMT_ANNUITY', 'AMT_GOODS_PRICE',
            'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS'
        ]
        
    def fit_transform(self, filepath):
        """
        Complete preprocessing pipeline.
        
        Returns:
            X: Scaled feature matrix (numpy array)
            y: Target variable
            features: List of feature names
        """
        # Load
        df = pd.read_csv(filepath)
        print(f"Loaded: {df.shape}")
        
        # Process
        df = self._create_derived_features(df)
        df = self._handle_known_anomaly(df)
        df = self._handle_missing_values(df)
        df = self._encode_categorical(df)
        
        # Select features
        X, y = self._select_features(df)
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Final validation: ensure no NaN
        if np.isnan(X_scaled).any():
            raise ValueError("NaN values detected after preprocessing!")
        
        print(f"Final shape: {X_scaled.shape}")
        print(f"Features: {len(self.feature_names)}")
        
        return X_scaled, y, self.feature_names
    
    def transform(self, df):
        """Transform new data using fitted preprocessor."""
        df = self._create_derived_features(df)
        df = self._handle_known_anomaly(df)
        df = self._handle_missing_values(df)
        df = self._encode_categorical(df, fit=False)
        X, _ = self._select_features(df)
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def _create_derived_features(self, df):
        """Feature engineering: ratios and transformations."""
        df = df.copy()
        
        # Age (from negative days)
        df['AGE'] = -df['DAYS_BIRTH'] / 365
        
        # Financial ratios (contextual features)
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        
        # Handle division by zero
        df['CREDIT_INCOME_RATIO'] = df['CREDIT_INCOME_RATIO'].replace([np.inf, -np.inf], np.nan)
        df['ANNUITY_INCOME_RATIO'] = df['ANNUITY_INCOME_RATIO'].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def _handle_known_anomaly(self, df):
        """Flag and remove known system bug (DAYS_EMPLOYED = 365243)."""
        df = df.copy()
        df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
        df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan
        return df
    
    def _handle_missing_values(self, df):
        """
        Critical Strategy: Missing as Signal
        1. Create binary flags
        2. Impute with mean (NOT -999 for neural nets)
        """
        df = df.copy()
        
        for col in self.missing_features:
            if col in df.columns and df[col].isna().any():
                # Binary flag
                df[f'{col}_missing'] = df[col].isna().astype(int)
                
                # Mean imputation
                df[col] = df[col].fillna(df[col].mean())
        
        # Handle any remaining NaN in ratios
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _encode_categorical(self, df, fit=True):
        """Minimal categorical encoding."""
        df = df.copy()
        
        categorical_features = {
            'NAME_INCOME_TYPE': 'income_encoded',
            'NAME_EDUCATION_TYPE': 'education_encoded'
        }
        
        for original, encoded in categorical_features.items():
            if original in df.columns:
                if fit:
                    self.encoders[original] = LabelEncoder()
                    df[encoded] = self.encoders[original].fit_transform(
                        df[original].fillna('Unknown')
                    )
                else:
                    df[encoded] = self.encoders[original].transform(
                        df[original].fillna('Unknown')
                    )
        
        return df
    
    def _select_features(self, df):
        """Select 20 key financial features."""
        self.feature_names = [
            # Financial amounts (4)
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
            
            # External scores + flags (4)
            'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'EXT_SOURCE_2_missing', 'EXT_SOURCE_3_missing',
            
            # Time features (4)
            'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH',
            
            # Derived ratios (2)
            'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO',
            
            # Demographics (3)
            'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'AGE',
            
            # Categorical (2)
            'income_encoded', 'education_encoded',
            
            # Anomaly flag (1)
            'DAYS_EMPLOYED_ANOM'
        ]
        
        X = df[self.feature_names].values
        y = df['TARGET'].values if 'TARGET' in df.columns else None
        
        return X, y


def get_feature_groups():
    """Return feature groups for analysis."""
    return {
        'financial': ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE'],
        'external_scores': ['EXT_SOURCE_2', 'EXT_SOURCE_3'],
        'temporal': ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'],
        'ratios': ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO'],
        'demographics': ['CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'AGE']
    }
