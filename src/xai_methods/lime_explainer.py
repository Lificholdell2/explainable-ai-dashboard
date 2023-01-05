import lime
import lime.lime_tabular
import numpy as np

class LimeExplainer:
    def __init__(self, model, feature_names, class_names, training_data):
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode=\'classification\'
        )
        self.model = model

    def explain_instance(self, instance, num_features=5):
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=self.model.predict_proba,
            num_features=num_features
        )
        return explanation.as_list()

# Change on 2023-01-03 09:44:15: security: Implement user authentication for dashboard access

# Change on 2023-01-05 11:49:35: fix: Resolve issues with interactive graph updates
