# src/utils.py
"""
Utility Functions for PCOS Detection App
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def create_metric_card(label, value, delta=None, help_text=""):
    """Create a metric card with optional delta"""
    st.metric(label=label, value=value, delta=delta, help=help_text)

def plot_confusion_matrix(cm, labels=['No PCOS', 'PCOS']):
    """Plot confusion matrix using Plotly"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        showscale=False
    ))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual'
    )
    return fig

def plot_feature_importance(features, importances, top_n=10):
    """Plot feature importance bar chart"""
    df = pd.DataFrame({
        'Feature': features[:top_n],
        'Importance': importances[:top_n]
    })
    
    fig = px.bar(
        df, x='Importance', y='Feature', orientation='h',
        title=f'Top {top_n} Feature Importance',
        color='Importance',
        color_continuous_scale='viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def plot_roc_curve(fpr, tpr, auc):
    """Plot ROC curve"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC (AUC = {auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    return fig

def plot_distribution(df, column, color_col='PCOS_Diagnosis'):
    """Plot distribution of a column by target"""
    fig = px.histogram(
        df, x=column, color=color_col,
        barmode='overlay',
        title=f'Distribution of {column}',
        opacity=0.7
    )
    return fig

def get_risk_level(probability):
    """Get risk level string from probability"""
    if probability < 0.3:
        return "Low Risk", "green"
    elif probability < 0.6:
        return "Moderate Risk", "orange"
    else:
        return "High Risk", "red"

def validate_input(value, min_val, max_val, input_type='number'):
    """Validate user input"""
    try:
        if input_type == 'number':
            return min_val <= float(value) <= max_val
        return True
    except:
        return False