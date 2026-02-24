# src/model.py
"""
Machine Learning Model for PCOS Detection
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import joblib
from pathlib import Path

class PCOSClassifier:
    """Main classifier for PCOS detection"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = self._get_model()
        self.is_trained = False
        
    def _get_model(self):
        """Initialize model based on type"""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        }
        return models.get(model_type, models['random_forest'])
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, self.predict_proba(X_test)[:, 1]),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics, y_pred
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """Load trained model from disk"""
        model = joblib.load(filepath)
        classifier = PCOSClassifier()
        classifier.model = model
        classifier.is_trained = True
        return classifier