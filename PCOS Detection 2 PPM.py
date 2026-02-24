# src/preprocessing.py
"""
Data Preprocessing Module for PCOS Detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class PCOSPreprocessor:
    """Handles all data preprocessing tasks"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'Age', 'Weight_kg', 'BMI', 'Cycle_R_I', 'Cycle_Length',
            'Follicle_No_L', 'Follicle_No_R', 'Avg_F_Size_L', 
            'Avg_F_Size_R', 'Endometrium_mm', 'FSH', 'LH', 'TSH', 'PRL', 'Vit_D', 'AMH'
        ]
        
    def clean_data(self, df):
        """Clean and validate data"""
        df_clean = df.copy()
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        
        # Remove outliers (IQR method) for some columns
        outlier_cols = ['BMI', 'Follicle_No_L', 'Follicle_No_R']
        for col in outlier_cols:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.05)
                Q3 = df_clean[col].quantile(0.95)
                df_clean[col] = df_clean[col].clip(Q1, Q3)
        
        return df_clean
    
    def prepare_features(self, df, target_col='PCOS_Diagnosis'):
        """Prepare features and target for modeling"""
        # Select only available features
        available_features = [f for f in self.feature_columns if f in df.columns]
        
        X = df[available_features]
        y = df[target_col]
        
        return X, y, available_features
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        return train_test_split(
            X, y, test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
    
    def scale_features(self, X_train, X_test):
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def get_feature_importance_data(self, model, feature_names):
        """Extract feature importance from model"""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        return {
            'features': [feature_names[i] for i in indices],
            'importances': importances[indices]
        }