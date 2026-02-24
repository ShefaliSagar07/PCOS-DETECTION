# src/data_generator.py
"""
Synthetic PCOS Dataset Generator
Creates realistic medical data for demonstration purposes
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_pcos_dataset(n_samples=1000, random_state=42):
    """
    Generate synthetic PCOS dataset with realistic correlations.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic PCOS data
    """
    np.random.seed(random_state)
    
    # Initialize data dictionary
    data = Demographics
    {}
    
    # data['Age'] = np.random.normal(28, 5, n_samples).clip(18, 45)
    data['Weight_kg'] = np.random.normal(65, 15, n_samples).clip(40, 120)
    data['Height_cm'] = np.random.normal(162, 8, n_samples).clip(145, 185)
    data['BMI'] = data['Weight_kg'] / ((data['Height_cm']/100) ** 2)
    
    # Menstrual Cycle Features
    data['Cycle_R_I'] = np.random.choice([2, 4], n_samples, p=[0.6, 0.4])  # 2=Regular, 4=Irregular
    data['Cycle_Length'] = np.random.normal(30, 5, n_samples).clip(21, 45)
    
    # Ultrasound Features (Key indicators for PCOS)
    data['Follicle_No_L'] = np.random.poisson(6, n_samples)
    data['Follicle_No_R'] = np.random.poisson(6, n_samples)
    data['Avg_F_Size_L'] = np.random.normal(12, 4, n_samples).clip(5, 25)
    data['Avg_F_Size_R'] = np.random.normal(12, 4, n_samples).clip(5, 25)
    data['Endometrium_mm'] = np.random.normal(9, 3, n_samples).clip(4, 18)
    
    # Additional Hormonal/Metabolic Features
    data['FSH'] = np.random.normal(6, 2, n_samples)
    data['LH'] = np.random.normal(5, 2, n_samples)
    data['TSH'] = np.random.normal(2.5, 1, n_samples)
    data['PRL'] = np.random.normal(15, 5, n_samples)
    data['Vit_D'] = np.random.normal(25, 10, n_samples)
    data['AMH'] = np.random.normal(4, 2, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable (PCOS diagnosis)
    # PCOS is associated with: 
    # - High follicle count (>12 total)
    # - Irregular cycles
    # - High BMI
    # - High AMH
    
    pcos_probability = (
        (df['Follicle_No_L'] + df['Follicle_No_R'] > 12) * 0.3 +
        (data['Cycle_R_I'] == 4) * 0.3 +
        (df['BMI'] > 28) * 0.2 +
        (df['AMH'] > 6) * 0.2
    )
    
    # Add noise and threshold
    noise = np.random.random(n_samples) * 0.2
    pcos_scores = pcos_probability + noise
    df['PCOS_Diagnosis'] = (pcos_scores > 0.45).astype(int)
    
    # Round values for realism
    df['Age'] = df['Age'].round(1)
    df['Weight_kg'] = df['Weight_kg'].round(1)
    df['Height_cm'] = df['Height_cm'].round(1)
    df['BMI'] = df['BMI'].round(2)
    df['Cycle_Length'] = df['Cycle_Length'].round(0).astype(int)
    df['Avg_F_Size_L'] = df['Avg_F_Size_L'].round(1)
    df['Avg_F_Size_R'] = df['Avg_F_Size_R'].round(1)
    df['Endometrium_mm'] = df['Endometrium_mm'].round(1)
    df['FSH'] = df['FSH'].round(2)
    df['LH'] = df['LH'].round(2)
    df['TSH'] = df['TSH'].round(2)
    df['PRL'] = df['PRL'].round(2)
    df['Vit_D'] = df['Vit_D'].round(2)
    df['AMH'] = df['AMH'].round(2)
    
    return df

def save_dataset(df, filepath='data/pcos_data.csv'):
    """Save dataset to CSV"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")

def load_or_generate_data(filepath='data/pcos_data.csv', n_samples=1000):
    """Load existing data or generate new"""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded existing dataset from {filepath}")
        return df
    except FileNotFoundError:
        print("Generating new synthetic dataset...")
        df = generate_pcos_dataset(n_samples)
        save_dataset(df, filepath)
        return df

if __name__ == "__main__":
    df = load_or_generate_data()
    print(f"Dataset shape: {df.shape}")
    print(f"PCOS positive cases: {df['PCOS_Diagnosis'].sum()} ({df['PCOS_Diagnosis'].mean()*100:.1f}%)")