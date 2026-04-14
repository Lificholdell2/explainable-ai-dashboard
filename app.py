import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Sample data (replace with actual data loading and XAI results)
df = pd.read_csv("data/sample_data.csv")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Explainable AI Dashboard"),

    html.Div([
        html.H2("Feature Importance"),
        dcc.Graph(id=\"feature-importance-graph\")
    ]),

    html.Div([
        html.H2("Individual Prediction Explanation"),
        dcc.Dropdown(
            id=\"sample-selector\",
            options=[{'label': str(i), 'value': i} for i in df.index],
            value=df.index[0]
        ),
        dcc.Graph(id="individual-explanation-graph")
    ])
])

@app.callback(
    Output("feature-importance-graph", "figure"),
    Input("sample-selector", "value") # Not directly used, but triggers update
)
def update_feature_importance(selected_sample):
    # In a real app, this would come from an XAI method (e.g., SHAP global importance)
    feature_importance = pd.DataFrame({
        'Feature': ['Feature A', 'Feature B', 'Feature C', 'Feature D'],
        'Importance': [0.4, 0.3, 0.2, 0.1]
    })
    fig = px.bar(feature_importance, x='Feature', y='Importance', title='Global Feature Importance')
    return fig

@app.callback(
    Output("individual-explanation-graph", "figure"),
    Input("sample-selector", "value")
)
def update_individual_explanation(selected_sample):
    # In a real app, this would come from an XAI method (e.g., LIME or SHAP for a single instance)
    sample_data = df.iloc[selected_sample]
    explanation_data = pd.DataFrame({
        'Feature': ['Feature A', 'Feature B', 'Feature C', 'Feature D'],
        'Contribution': [0.1, -0.05, 0.2, 0.03]
    })
    fig = px.bar(explanation_data, x='Feature', y='Contribution', title=f'Explanation for Sample {selected_sample}')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
