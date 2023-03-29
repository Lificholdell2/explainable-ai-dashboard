import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.xai_methods.lime_explainer import LimeExplainer
from src.xai_methods.shap_explainer import ShapExplainer
from src.visualizations.dashboard_components import create_feature_importance_bar_chart, create_individual_explanation_waterfall_chart

# Load sample data and train a simple model
df = pd.read_csv("data/sample_data.csv")
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

feature_names = X.columns.tolist()
class_names = [str(c) for c in model.classes_]

# Initialize XAI explainers
lime_explainer = LimeExplainer(model.predict_proba, feature_names, class_names, X_train.values)
shap_explainer = ShapExplainer(model, X_train)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Explainable AI Dashboard"),

    html.Div([
        html.H2("Global Feature Importance"),
        dcc.Graph(id="feature-importance-graph")
    ]),

    html.Div([
        html.H2("Individual Prediction Explanation"),
        html.Label("Select Sample:"),
        dcc.Dropdown(
            id="sample-selector",
            options=[{'label': f'Sample {i}', 'value': i} for i in X_test.index],
            value=X_test.index[0]
        ),
        dcc.Graph(id="individual-explanation-graph")
    ])
])

@app.callback(
    Output("feature-importance-graph", "figure"),
    Input("sample-selector", "value") # Trigger on any input change, but global importance is static
)
def update_feature_importance(selected_sample_index):
    # Using SHAP for global feature importance (mean absolute SHAP values)
    shap_values = shap_explainer.explainer(X_train)
    mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
    feature_importance = dict(zip(feature_names, mean_abs_shap_values))
    return create_feature_importance_bar_chart(feature_importance, title="Global Feature Importance (Mean |SHAP|)")

@app.callback(
    Output("individual-explanation-graph", "figure"),
    Input("sample-selector", "value")
)
def update_individual_explanation(selected_sample_index):
    if selected_sample_index is None:
        return go.Figure()

    instance = X_test.loc[selected_sample_index].values
    
    # Using LIME for individual explanation
    lime_explanation = lime_explainer.explain_instance(instance)
    
    # Convert LIME explanation to a dictionary for the waterfall chart
    explanation_dict = {feature: value for feature, value in lime_explanation}

    return create_individual_explanation_waterfall_chart(explanation_dict, title=f'LIME Explanation for Sample {selected_sample_index}')

if __name__ == '__main__':
    app.run_server(debug=True)

# Change on 2023-01-02 13:42:22: chore: Clean up unused visualization assets

# Change on 2023-01-16 14:28:21: docs: Create detailed API documentation for XAI components

# Change on 2023-01-19 11:51:14: chore: Upgrade Dash and Plotly dependencies

# Change on 2023-01-24 11:02:46: security: Implement user authentication for dashboard access

# Change on 2023-01-25 14:54:13: perf: Optimize visualization rendering for large datasets

# Change on 2023-02-02 17:27:27: refactor: Improve dashboard layout and responsiveness

# Change on 2023-02-03 16:25:33: fix: Resolve issues with interactive graph updates

# Change on 2023-02-06 12:28:16: test: Add unit tests for XAI explanation generation

# Change on 2023-02-07 09:17:17: feat: Add support for new XAI methods (e.g., Grad-CAM)

# Change on 2023-02-10 09:04:49: refactor: Improve dashboard layout and responsiveness

# Change on 2023-02-13 13:42:16: style: Apply consistent styling to dashboard components

# Change on 2023-02-14 13:48:06: refactor: Modularize XAI method implementations

# Change on 2023-02-14 17:03:39: refactor: Improve dashboard layout and responsiveness

# Change on 2023-02-15 15:03:20: build: Set up Dockerfile for dashboard deployment

# Change on 2023-02-17 12:40:34: refactor: Modularize XAI method implementations

# Change on 2023-02-21 09:33:00: security: Implement user authentication for dashboard access

# Change on 2023-02-28 09:23:52: fix: Correct data loading and preprocessing for XAI models

# Change on 2023-03-14 11:59:15: chore: Clean up unused visualization assets

# Change on 2023-03-16 12:42:09: refactor: Improve dashboard layout and responsiveness

# Change on 2023-03-21 10:52:43: perf: Optimize visualization rendering for large datasets

# Change on 2023-03-24 11:08:27: perf: Optimize visualization rendering for large datasets

# Change on 2023-03-28 13:26:21: fix: Resolve issues with interactive graph updates

# Change on 2023-03-29 16:48:44: feat: Integrate LIME and SHAP explainers into the dashboard

# Change on 2023-03-29 13:24:27: build: Set up Dockerfile for dashboard deployment
