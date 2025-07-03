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

# Change on 2023-06-22 14:57:44: refactor: Improve dashboard layout and responsiveness

# Change on 2023-06-29 13:38:46: fix: Correct data loading and preprocessing for XAI models

# Change on 2023-07-04 09:54:11: perf: Optimize visualization rendering for large datasets

# Change on 2023-07-10 13:58:46: feat: Integrate LIME and SHAP explainers into the dashboard

# Change on 2023-07-12 15:21:31: refactor: Modularize XAI method implementations

# Change on 2023-07-17 15:24:29: build: Set up Dockerfile for dashboard deployment

# Change on 2023-07-25 12:06:03: test: Add unit tests for XAI explanation generation

# Change on 2023-07-28 16:53:59: chore: Upgrade Dash and Plotly dependencies

# Change on 2023-08-02 12:32:28: refactor: Improve dashboard layout and responsiveness

# Change on 2023-08-03 15:34:27: feat: Integrate LIME and SHAP explainers into the dashboard

# Change on 2023-08-08 13:42:39: test: Add unit tests for XAI explanation generation

# Change on 2023-08-09 12:59:26: docs: Create detailed API documentation for XAI components

# Change on 2023-08-09 10:50:49: refactor: Improve dashboard layout and responsiveness

# Change on 2023-09-05 10:07:12: fix: Correct data loading and preprocessing for XAI models

# Change on 2023-09-07 13:51:30: fix: Correct data loading and preprocessing for XAI models

# Change on 2023-09-08 15:27:13: security: Implement user authentication for dashboard access

# Change on 2023-09-13 15:41:54: docs: Create detailed API documentation for XAI components

# Change on 2023-09-14 09:51:53: refactor: Improve dashboard layout and responsiveness

# Change on 2023-09-18 10:45:55: chore: Clean up unused visualization assets

# Change on 2023-09-19 13:48:06: refactor: Improve dashboard layout and responsiveness

# Change on 2023-09-26 16:39:55: docs: Update usage instructions for running the XAI dashboard

# Change on 2023-10-03 16:13:24: build: Set up Dockerfile for dashboard deployment

# Change on 2023-10-10 14:18:53: test: Add unit tests for XAI explanation generation

# Change on 2023-10-12 10:01:52: chore: Upgrade Dash and Plotly dependencies

# Change on 2023-10-23 10:39:56: fix: Correct data loading and preprocessing for XAI models

# Change on 2023-10-26 16:56:24: refactor: Improve dashboard layout and responsiveness

# Change on 2023-10-26 16:25:57: chore: Upgrade Dash and Plotly dependencies

# Change on 2023-11-01 15:32:22: perf: Optimize visualization rendering for large datasets

# Change on 2023-11-03 13:00:23: refactor: Improve dashboard layout and responsiveness

# Change on 2023-11-06 15:02:47: build: Set up Dockerfile for dashboard deployment

# Change on 2023-11-07 17:07:55: docs: Create detailed API documentation for XAI components

# Change on 2023-11-08 15:51:41: docs: Update usage instructions for running the XAI dashboard

# Change on 2023-11-15 10:29:44: build: Set up Dockerfile for dashboard deployment

# Change on 2023-11-22 16:47:10: chore: Upgrade Dash and Plotly dependencies

# Change on 2023-11-24 12:58:44: feat: Add support for new XAI methods (e.g., Grad-CAM)

# Change on 2023-12-05 15:37:40: docs: Update usage instructions for running the XAI dashboard

# Change on 2023-12-05 10:13:59: docs: Update usage instructions for running the XAI dashboard

# Change on 2023-12-06 12:21:09: feat: Add support for new XAI methods (e.g., Grad-CAM)

# Change on 2023-12-11 11:30:28: feat: Integrate LIME and SHAP explainers into the dashboard

# Change on 2023-12-11 11:48:23: chore: Upgrade Dash and Plotly dependencies

# Change on 2023-12-12 12:14:00: feat: Add support for new XAI methods (e.g., Grad-CAM)

# Change on 2023-12-19 10:10:07: build: Set up Dockerfile for dashboard deployment

# Change on 2023-12-21 16:05:26: docs: Create detailed API documentation for XAI components

# Change on 2023-12-25 10:14:42: chore: Clean up unused visualization assets

# Change on 2023-12-27 13:58:09: fix: Resolve issues with interactive graph updates

# Change on 2024-01-11 12:42:30: refactor: Modularize XAI method implementations

# Change on 2024-01-15 16:55:03: fix: Correct data loading and preprocessing for XAI models

# Change on 2024-01-16 11:18:11: perf: Optimize visualization rendering for large datasets

# Change on 2024-01-17 09:34:20: docs: Create detailed API documentation for XAI components

# Change on 2024-01-18 15:20:18: fix: Resolve issues with interactive graph updates

# Change on 2024-01-25 16:18:50: perf: Optimize visualization rendering for large datasets

# Change on 2024-01-25 11:24:59: feat: Add support for new XAI methods (e.g., Grad-CAM)

# Change on 2024-01-31 17:03:10: feat: Integrate LIME and SHAP explainers into the dashboard

# Change on 2024-02-02 16:31:49: perf: Optimize visualization rendering for large datasets

# Change on 2024-02-08 12:52:00: chore: Upgrade Dash and Plotly dependencies

# Change on 2024-02-19 10:14:00: feat: Add support for new XAI methods (e.g., Grad-CAM)

# Change on 2024-02-20 14:25:46: test: Add unit tests for XAI explanation generation

# Change on 2024-03-04 11:30:52: docs: Create detailed API documentation for XAI components

# Change on 2024-03-04 11:53:52: build: Set up Dockerfile for dashboard deployment

# Change on 2024-03-05 17:30:50: refactor: Improve dashboard layout and responsiveness

# Change on 2024-03-08 09:46:08: perf: Optimize visualization rendering for large datasets

# Change on 2024-03-08 16:40:56: fix: Resolve issues with interactive graph updates

# Change on 2024-04-09 16:56:38: chore: Upgrade Dash and Plotly dependencies

# Change on 2024-04-15 13:50:04: style: Apply consistent styling to dashboard components

# Change on 2024-04-18 13:47:03: feat: Add support for new XAI methods (e.g., Grad-CAM)

# Change on 2024-05-07 13:50:12: feat: Add support for new XAI methods (e.g., Grad-CAM)

# Change on 2024-05-10 15:50:09: test: Add unit tests for XAI explanation generation

# Change on 2024-05-13 09:48:25: docs: Create detailed API documentation for XAI components

# Change on 2024-05-14 12:09:18: docs: Create detailed API documentation for XAI components

# Change on 2024-05-21 11:49:25: chore: Upgrade Dash and Plotly dependencies

# Change on 2024-05-22 11:49:38: perf: Optimize visualization rendering for large datasets

# Change on 2024-05-23 16:26:56: docs: Update usage instructions for running the XAI dashboard

# Change on 2024-05-30 09:46:29: feat: Integrate LIME and SHAP explainers into the dashboard

# Change on 2024-06-11 13:16:59: refactor: Modularize XAI method implementations

# Change on 2024-06-11 12:29:37: build: Set up Dockerfile for dashboard deployment

# Change on 2024-06-12 16:23:32: fix: Correct data loading and preprocessing for XAI models

# Change on 2024-06-17 10:29:25: style: Apply consistent styling to dashboard components

# Change on 2024-06-25 11:38:30: docs: Update usage instructions for running the XAI dashboard

# Change on 2024-06-26 16:51:16: refactor: Improve dashboard layout and responsiveness

# Change on 2024-06-26 11:52:00: docs: Create detailed API documentation for XAI components

# Change on 2024-06-27 13:07:37: docs: Create detailed API documentation for XAI components

# Change on 2024-07-01 12:09:07: docs: Update usage instructions for running the XAI dashboard

# Change on 2024-07-04 09:53:17: refactor: Modularize XAI method implementations

# Change on 2024-07-10 16:17:09: perf: Optimize visualization rendering for large datasets

# Change on 2024-07-15 16:39:32: chore: Clean up unused visualization assets

# Change on 2024-07-15 15:02:42: fix: Resolve issues with interactive graph updates

# Change on 2024-07-16 13:17:34: chore: Clean up unused visualization assets

# Change on 2024-07-19 11:45:53: security: Implement user authentication for dashboard access

# Change on 2024-07-24 16:15:56: docs: Create detailed API documentation for XAI components

# Change on 2024-07-26 11:35:49: fix: Correct data loading and preprocessing for XAI models

# Change on 2024-07-26 12:58:18: fix: Correct data loading and preprocessing for XAI models

# Change on 2024-07-31 16:34:22: refactor: Modularize XAI method implementations

# Change on 2024-08-02 17:36:09: fix: Resolve issues with interactive graph updates

# Change on 2024-08-08 14:50:54: fix: Correct data loading and preprocessing for XAI models

# Change on 2024-08-16 11:13:21: fix: Resolve issues with interactive graph updates

# Change on 2024-08-16 13:38:50: test: Add unit tests for XAI explanation generation

# Change on 2024-08-21 13:24:52: chore: Clean up unused visualization assets

# Change on 2024-08-23 13:22:55: style: Apply consistent styling to dashboard components

# Change on 2024-08-30 13:16:07: refactor: Improve dashboard layout and responsiveness

# Change on 2024-09-02 17:55:07: chore: Upgrade Dash and Plotly dependencies

# Change on 2024-09-04 16:06:42: build: Set up Dockerfile for dashboard deployment

# Change on 2024-09-11 13:18:19: refactor: Improve dashboard layout and responsiveness

# Change on 2024-09-13 14:48:39: test: Add unit tests for XAI explanation generation

# Change on 2024-09-16 11:58:29: docs: Update usage instructions for running the XAI dashboard

# Change on 2024-09-17 16:40:09: build: Set up Dockerfile for dashboard deployment

# Change on 2024-09-26 13:47:22: test: Add unit tests for XAI explanation generation

# Change on 2024-10-07 11:30:20: security: Implement user authentication for dashboard access

# Change on 2024-10-11 09:06:30: fix: Correct data loading and preprocessing for XAI models

# Change on 2024-10-14 15:30:32: perf: Optimize visualization rendering for large datasets

# Change on 2024-10-31 09:36:22: security: Implement user authentication for dashboard access

# Change on 2024-11-04 10:43:16: chore: Clean up unused visualization assets

# Change on 2024-11-08 10:46:06: security: Implement user authentication for dashboard access

# Change on 2024-11-13 15:51:49: docs: Create detailed API documentation for XAI components

# Change on 2024-11-15 15:41:21: refactor: Improve dashboard layout and responsiveness

# Change on 2024-12-12 12:40:38: chore: Clean up unused visualization assets

# Change on 2024-12-17 09:20:31: build: Set up Dockerfile for dashboard deployment

# Change on 2024-12-18 14:04:49: chore: Clean up unused visualization assets

# Change on 2024-12-19 17:48:06: docs: Create detailed API documentation for XAI components

# Change on 2024-12-24 16:16:57: test: Add unit tests for XAI explanation generation

# Change on 2025-01-02 12:23:35: build: Set up Dockerfile for dashboard deployment

# Change on 2025-01-02 16:54:22: feat: Add support for new XAI methods (e.g., Grad-CAM)

# Change on 2025-01-14 10:56:12: build: Set up Dockerfile for dashboard deployment

# Change on 2025-01-31 12:02:48: refactor: Improve dashboard layout and responsiveness

# Change on 2025-02-10 10:06:28: docs: Create detailed API documentation for XAI components

# Change on 2025-02-14 09:35:34: refactor: Improve dashboard layout and responsiveness

# Change on 2025-02-18 13:47:19: chore: Upgrade Dash and Plotly dependencies

# Change on 2025-02-20 09:42:41: feat: Integrate LIME and SHAP explainers into the dashboard

# Change on 2025-02-24 10:39:17: feat: Add support for new XAI methods (e.g., Grad-CAM)

# Change on 2025-03-04 14:26:28: fix: Resolve issues with interactive graph updates

# Change on 2025-03-06 12:57:25: security: Implement user authentication for dashboard access

# Change on 2025-03-10 15:28:25: build: Set up Dockerfile for dashboard deployment

# Change on 2025-03-12 13:19:31: chore: Upgrade Dash and Plotly dependencies

# Change on 2025-03-14 10:36:01: chore: Clean up unused visualization assets

# Change on 2025-03-18 11:57:43: fix: Resolve issues with interactive graph updates

# Change on 2025-03-21 14:39:26: refactor: Modularize XAI method implementations

# Change on 2025-03-31 14:09:23: docs: Update usage instructions for running the XAI dashboard

# Change on 2025-04-01 14:06:47: docs: Create detailed API documentation for XAI components

# Change on 2025-04-01 12:40:45: docs: Create detailed API documentation for XAI components

# Change on 2025-04-23 13:08:46: refactor: Modularize XAI method implementations

# Change on 2025-04-29 09:14:49: fix: Resolve issues with interactive graph updates

# Change on 2025-04-29 11:47:09: perf: Optimize visualization rendering for large datasets

# Change on 2025-05-01 14:33:01: test: Add unit tests for XAI explanation generation

# Change on 2025-05-05 14:54:27: feat: Integrate LIME and SHAP explainers into the dashboard

# Change on 2025-05-06 09:47:33: security: Implement user authentication for dashboard access

# Change on 2025-05-08 13:25:42: fix: Correct data loading and preprocessing for XAI models

# Change on 2025-05-09 13:57:47: test: Add unit tests for XAI explanation generation

# Change on 2025-05-13 14:22:48: feat: Integrate LIME and SHAP explainers into the dashboard

# Change on 2025-05-14 16:24:19: chore: Clean up unused visualization assets

# Change on 2025-05-15 17:52:34: refactor: Improve dashboard layout and responsiveness

# Change on 2025-05-23 15:07:50: refactor: Improve dashboard layout and responsiveness

# Change on 2025-06-17 10:01:54: refactor: Modularize XAI method implementations

# Change on 2025-06-19 16:03:26: security: Implement user authentication for dashboard access

# Change on 2025-06-25 16:52:54: refactor: Modularize XAI method implementations

# Change on 2025-06-30 10:34:07: chore: Clean up unused visualization assets

# Change on 2025-07-02 11:07:20: refactor: Improve dashboard layout and responsiveness

# Change on 2025-07-03 09:54:30: security: Implement user authentication for dashboard access
