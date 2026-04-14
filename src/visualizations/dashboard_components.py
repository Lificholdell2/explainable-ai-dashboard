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
