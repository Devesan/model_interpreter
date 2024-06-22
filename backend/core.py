# backend/core.py

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def load_model(file):
    return joblib.load(file)


def identify_model(model):
    # Dummy function to simulate model identification
    # Replace with actual model identification logic
    model_info = {
        "type": "RandomForestClassifier",
        "columns": ["feature1", "feature2", "feature3"],
        "complexity": "Medium"
    }
    return model_info


def process_input_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    else:
        st.error("Unsupported file type.")
        return None


def predict(model, data):
    predictions = model.predict(data)
    return predictions


def explain_predictions(model, data, predictions):
    explanations = []
    for i, prediction in enumerate(predictions):
        explanation = f"Record {i + 1}: Feature1 contributed X, Feature2 contributed Y, etc."
        explanations.append(explanation)
    return explanations


def display_feature_importance(model, columns):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df.head(5), x='importance', y='feature')
    plt.title('Top 5 Feature Importances')
    st.pyplot(plt)
