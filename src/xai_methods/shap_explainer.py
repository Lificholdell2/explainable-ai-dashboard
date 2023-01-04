import shap

class ShapExplainer:
    def __init__(self, model, data):
        self.explainer = shap.Explainer(model, data)

    def explain_instance(self, instance):
        shap_values = self.explainer(instance)
        return shap_values

# Change on 2023-01-02 14:32:05: refactor: Modularize XAI method implementations

# Change on 2023-01-04 12:00:09: feat: Add support for new XAI methods (e.g., Grad-CAM)
