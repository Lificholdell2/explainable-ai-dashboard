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

# Change on 2023-02-13 12:27:19: chore: Clean up unused visualization assets

# Change on 2023-02-13 09:17:04: docs: Create detailed API documentation for XAI components

# Change on 2023-02-20 13:23:46: chore: Upgrade Dash and Plotly dependencies

# Change on 2023-02-21 15:44:49: test: Add unit tests for XAI explanation generation

# Change on 2023-02-24 12:07:44: fix: Resolve issues with interactive graph updates

# Change on 2023-03-09 16:02:44: chore: Clean up unused visualization assets

# Change on 2023-03-14 14:09:19: style: Apply consistent styling to dashboard components

# Change on 2023-03-14 13:33:50: docs: Create detailed API documentation for XAI components

# Change on 2023-03-27 12:08:44: docs: Create detailed API documentation for XAI components

# Change on 2023-03-29 11:22:27: docs: Create detailed API documentation for XAI components

# Change on 2023-04-18 16:55:23: feat: Integrate LIME and SHAP explainers into the dashboard

# Change on 2023-04-21 16:48:15: feat: Add support for new XAI methods (e.g., Grad-CAM)

# Change on 2023-05-04 10:44:14: refactor: Modularize XAI method implementations

# Change on 2023-05-05 13:13:55: chore: Upgrade Dash and Plotly dependencies

# Change on 2023-05-11 17:57:00: style: Apply consistent styling to dashboard components

# Change on 2023-05-19 15:34:02: chore: Clean up unused visualization assets

# Change on 2023-05-19 09:38:40: fix: Resolve issues with interactive graph updates

# Change on 2023-05-24 13:34:32: refactor: Improve dashboard layout and responsiveness

# Change on 2023-05-30 16:19:47: test: Add unit tests for XAI explanation generation

# Change on 2023-06-19 17:09:10: feat: Integrate LIME and SHAP explainers into the dashboard
