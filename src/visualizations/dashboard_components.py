import plotly.graph_objects as go
import pandas as pd

def create_feature_importance_bar_chart(feature_importances, title="Feature Importance"):
    df = pd.DataFrame(list(feature_importances.items()), columns=["Feature", "Importance"])
    fig = go.Figure(data=[go.Bar(x=df["Feature"], y=df["Importance"])])
    fig.update_layout(title_text=title)
    return fig

def create_individual_explanation_waterfall_chart(explanation_values, title="Individual Prediction Explanation"):
    # explanation_values is expected to be a list of (feature, value) tuples
    features = [item[0] for item in explanation_values]
    values = [item[1] for item in explanation_values]

    fig = go.Figure(go.Waterfall(
        name = "Explanation",
        orientation = "v",
        measure = ["relative"] * len(values),
        x = features,
        textposition = "outside",
        text = [f"{v:.2f}" for v in values],
        y = values,
        connector = {"line": {"color": "rgb(63, 63, 63)"}},
    ))

    fig.update_layout(title_text=title, showlegend = True)
    return fig

# Change on 2023-01-04 16:57:10: feat: Add support for new XAI methods (e.g., Grad-CAM)

# Change on 2023-01-05 14:59:01: build: Set up Dockerfile for dashboard deployment

# Change on 2023-01-12 16:22:52: feat: Integrate LIME and SHAP explainers into the dashboard

# Change on 2023-01-19 09:21:29: feat: Integrate LIME and SHAP explainers into the dashboard

# Change on 2023-01-19 14:39:34: chore: Clean up unused visualization assets

# Change on 2023-01-24 17:15:39: feat: Add support for new XAI methods (e.g., Grad-CAM)

# Change on 2023-01-30 14:27:57: build: Set up Dockerfile for dashboard deployment

# Change on 2023-02-01 14:00:28: docs: Update usage instructions for running the XAI dashboard

# Change on 2023-02-07 15:03:55: build: Set up Dockerfile for dashboard deployment
