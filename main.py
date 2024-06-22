# main.py

import streamlit as st
from backend.core import load_model, identify_model, process_input_file, predict, explain_predictions, \
    display_feature_importance


def main():
    st.title("Model Prediction and Explanation Chatbot")

    uploaded_model = st.file_uploader("Upload your model", type=["pkl", "joblib"])
    if uploaded_model:
        model = load_model(uploaded_model)
        model_info = identify_model(model)
        st.write(f"Model Type: {model_info['type']}")
        st.write(f"Columns: {', '.join(model_info['columns'])}")
        st.write(f"Complexity: {model_info['complexity']}")

        input_file = st.file_uploader("Upload input CSV or JSON ", type=["csv", "json"])
        if input_file:
            input_data = process_input_file(input_file)
            if input_data is not None:
                st.write("Input Data:")
                st.write(input_data)
                if st.button("Predict"):
                    predictions = predict(model, input_data.drop(['Unnamed: 0'],axis=1))
                    explanations = explain_predictions(model, input_data.drop(['Unnamed: 0'],axis=1), predictions)
                    input_data['predictions'] = predictions
                    input_data['explanations'] = explanations

                    st.write("Predictions and Explanations:")
                    st.write(input_data)
                    display_feature_importance(model, model_info['columns'])

                    csv = input_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Predictions",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv'
                    )


if __name__ == "__main__":
    main()
