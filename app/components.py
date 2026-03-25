"""
components.py - Reusable Chart & Visualization Components
==========================================================
Provides helper functions to create interactive Plotly charts
and other visual components for the Streamlit dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def create_gauge_chart(value: float, title: str, color_thresholds=None):
    """
    Creates a beautiful gauge chart for displaying metrics like accuracy.

    Args:
        value: Value between 0 and 100.
        title: Chart title.
        color_thresholds: Optional list of (threshold, color) tuples.
    """
    if color_thresholds is None:
        color_thresholds = [
            (50, "#ff4444"),   # Red < 50%
            (70, "#ffaa00"),   # Orange < 70%
            (85, "#00cc66"),   # Green < 85%
            (100, "#00ff88")   # Bright green >= 85%
        ]

    # Determine color based on value
    bar_color = color_thresholds[-1][1]
    for threshold, color in color_thresholds:
        if value <= threshold:
            bar_color = color
            break

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={'suffix': '%', 'font': {'size': 40, 'color': '#333333'}},
        title={'text': title, 'font': {'size': 18, 'color': '#555555'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#555555'},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': '#f8f9fa',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 68, 68, 0.1)'},
                {'range': [50, 70], 'color': 'rgba(255, 170, 0, 0.1)'},
                {'range': [70, 85], 'color': 'rgba(0, 204, 102, 0.1)'},
                {'range': [85, 100], 'color': 'rgba(0, 255, 136, 0.1)'},
            ],
            'threshold': {
                'line': {'color': '#ffffff', 'width': 2},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


def create_feature_importance_chart(fi_df: pd.DataFrame):
    """
    Creates a horizontal bar chart showing feature importance.

    Args:
        fi_df: DataFrame with 'Feature' and 'Importance' columns.
    """
    fi_sorted = fi_df.sort_values('Importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=fi_sorted['Importance'],
        y=fi_sorted['Feature'],
        orientation='h',
        marker=dict(
            color=fi_sorted['Importance'],
            colorscale='Tealgrn',
            line=dict(color='rgba(0,255,136,0.3)', width=1)
        ),
        text=[f'{v:.3f}' for v in fi_sorted['Importance']],
        textposition='outside',
        textfont=dict(color='#555555', size=12)
    ))

    fig.update_layout(
        title=dict(
            text='🏆 Feature Importance Ranking',
            font=dict(size=20, color='#333333')
        ),
        xaxis=dict(
            title=dict(text='Importance Score', font=dict(color='#555555')),
            tickfont=dict(color='#777777'),
            gridcolor='rgba(0,0,0,0.05)'
        ),
        yaxis=dict(
            tickfont=dict(color='#333333', size=13)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(l=10, r=60, t=60, b=40)
    )

    return fig


def create_confusion_matrix_chart(cm: np.ndarray, labels=None):
    """
    Creates an annotated heatmap for the confusion matrix.

    Args:
        cm: 2x2 confusion matrix array.
        labels: Class labels.
    """
    if labels is None:
        labels = ['Safe (0)', 'Outbreak (1)']

    # Normalize for color intensity
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    annotations = []
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            annotations.append(
                dict(
                    x=labels[j], y=labels[i],
                    text=f'{cm[i][j]:,}<br>({cm_normalized[i][j]:.1%})',
                    showarrow=False,
                    font=dict(size=16, color='white' if cm_normalized[i][j] > 0.5 else '#333333')
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=labels,
        y=labels,
        colorscale='Tealgrn',
        showscale=False,
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{customdata:,}<extra></extra>',
        customdata=cm
    ))

    fig.update_layout(
        title=dict(
            text='📊 Confusion Matrix',
            font=dict(size=20, color='#333333')
        ),
        xaxis=dict(title=dict(text='Predicted', font=dict(color='#555555')), tickfont=dict(color='#333333')),
        yaxis=dict(title=dict(text='Actual', font=dict(color='#555555')), tickfont=dict(color='#333333'), autorange='reversed'),
        annotations=annotations,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=380,
        margin=dict(l=20, r=20, t=60, b=40)
    )

    return fig


def create_outbreak_distribution_chart(y: pd.Series):
    """
    Creates a donut chart showing the Safe vs Outbreak distribution.
    """
    counts = y.value_counts()
    labels = ['Safe (No Disease)', 'Outbreak Risk']
    values = [counts.get(0, 0), counts.get(1, 0)]
    colors = ['#00cc66', '#ff4444']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color='#ffffff', width=3)),
        textfont=dict(size=14, color='#ffffff'),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])

    fig.update_layout(
        title=dict(
            text='🎯 Target Distribution',
            font=dict(size=20, color='#333333')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color='#555555')),
        height=380,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def create_prediction_result_card(prediction: int, probability: float):
    """
    Returns styled HTML for the prediction result display.
    """
    if prediction == 1:
        icon = "🚨"
        status = "OUTBREAK RISK DETECTED"
        color = "#ff4444"
        bg = "rgba(255, 68, 68, 0.15)"
        border = "rgba(255, 68, 68, 0.4)"
        advice = "⚠️ Immediate intervention recommended. Check water treatment systems and deploy medical resources."
    else:
        icon = "✅"
        status = "AREA IS SAFE"
        color = "#00cc66"
        bg = "rgba(0, 204, 102, 0.15)"
        border = "rgba(0, 204, 102, 0.4)"
        advice = "👍 No immediate risk detected. Continue routine monitoring."

    confidence = probability * 100 if prediction == 1 else (1 - probability) * 100

    html = f"""
    <div style="
        background: {bg};
        border: 2px solid {border};
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    ">
        <div style="font-size: 48px; margin-bottom: 10px;">{icon}</div>
        <h2 style="color: {color}; margin: 0; font-size: 28px;">{status}</h2>
        <p style="color: #666666; font-size: 16px; margin-top: 10px;">
            Confidence: <strong style="color: {color};">{confidence:.1f}%</strong>
        </p>
        <p style="color: #444444; font-size: 14px; margin-top: 15px;
                  background: rgba(0,0,0,0.05); padding: 12px; border-radius: 8px;">
            {advice}
        </p>
    </div>
    """
    return html
