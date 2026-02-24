# app.py
"""
PCOS Detection Assistant - Main Application
============================================
A comprehensive machine learning application for detecting Polycystic Ovary Syndrome (PCOS).
Built with Streamlit, Scikit-Learn, and Plotly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings

# Import custom modules
from src.data_generator import load_or_generate_data
from src.preprocessing import PCOSPreprocessor
from src.model import PCOSClassifier
from src.utils import (
    create_metric_card, plot_confusion_matrix, 
    plot_feature_importance, get_risk_level
)

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="PCOS Detection Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
    .risk-moderate {
        color: #f39c12;
        font-weight: bold;
    }
    .risk-low {