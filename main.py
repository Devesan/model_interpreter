import streamlit as st
from backend.core import load_model, load_and_explain, generate_descriptions, process_input_file
import json


def main():
    st.title("Model Interpreter")

    # Step 1: Get business context
    business_context = st.text_input('Please give a brief description of the business context of the model')

    if business_context:
        # Step 2: Choose workflow
        choice = st.selectbox('What would you like to do?',
                              ('Generate SHAP and Description', 'Just Generate Description'))

        if choice == 'Generate SHAP and Description':
            # Upload model
            uploaded_model = st.file_uploader("Upload your model", type=["pkl", "joblib"])
            if uploaded_model:
                file, model = load_model(uploaded_model)
                st.write(f"Uploaded model {type(model)} from {file}")
                st.session_state['model_file'] = file
                st.session_state['model'] = model

            if 'model' in st.session_state:
                model = st.session_state['model']
                file = st.session_state['model_file']
                # Upload input data
                input_file = st.file_uploader("Upload input CSV or JSON ", type=["csv", "json"])
                if input_file and 'input_data' not in st.session_state:
                    input_data = process_input_file(input_file)
                    if input_data is not None:
                        st.write("Input Data:")
                        st.write(input_data)
                        st.session_state['input_data'] = input_data
                        st.session_state['input_file'] = input_file.name
                        print('Input data added')

                if 'input_data' in st.session_state:
                    input_data = st.session_state['input_data']
                    st.write("Input Data:")
                    st.write(input_data)

                    # Generate SHAP values
                    if st.button("Generate SHAP Values"):
                        result = load_and_explain(input_file, file, model)
                        # result = {'output':'done'}
                        st.write("SHAP Values Generated:\n")
                        st.write(result['output'])
                        if 'Error' not in result['output']:
                            st.session_state['shap_json'] = 'shap_values.json'

        # Generate descriptions
        if 'shap_json' in st.session_state:
            if st.button("Generate Descriptions"):
                with open(st.session_state['shap_json'], 'r') as f:
                    shap_json = json.load(f)
                overall_summary = generate_descriptions(shap_json, business_context)
                st.write("Descriptions Generated:\n")
                st.write(overall_summary)

        elif choice == 'Just Generate Description':
            st.write(
                "Please ensure your SHAP data is in a JSON format with keys 'shap_values' and 'feature_names'. The file name should be 'shap_values.json'.")

            # Upload SHAP JSON
            shap_file = st.file_uploader("Upload SHAP values JSON", type=["json"])
            if shap_file:
                shap_json = json.load(shap_file)
                if 'shap_values' in shap_json and 'feature_names' in shap_json:
                    st.write("SHAP values and feature names loaded.")
                    if st.button("Generate Descriptions"):
                        overall_summary = generate_descriptions(shap_json,
                                                                business_context)
                        st.write("Descriptions Generated:\n")
                        st.write(overall_summary)
                else:
                    st.error(
                        "Invalid JSON format. Please ensure the file contains 'shap_values' and 'feature_names' keys.")


if __name__ == "__main__":
    main()
