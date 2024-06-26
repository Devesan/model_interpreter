import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Any, List
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool, tool, BaseTool
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from langchain.chains import LLMChain
import json
load_dotenv()

def load_model(file):
    return file.name, joblib.load(file)

def load_and_explain(input_file, file, model):
    # Initialize the LLM agent with Python REPL capabilities
    instructions = """
        You are an agent designed to write and execute python code to answer questions.
        You have access to a python REPL, which you can use to execute python code. If you get an error, debug your code and try again.
        Only use the output of your code to answer the question. You might know the answer without running any code, but you should still run the code to get the answer.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt_py = hub.pull("langchain-ai/react-agent-template")
    python_prompt = base_prompt_py.partial(instructions=instructions)
    python_tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=python_prompt,
        llm=ChatOpenAI(temperature=0.2, model="gpt-4-turbo"),
        tools=python_tools
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=python_tools, verbose=True)

    def python_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

        # Step 1: Generate SHAP Values and Save to CSV
    shap_values_prompt = f"""
                You are given a {type(model)} model, which you can load from the file named {file} using joblib. You also have an input file containing test data of type {input_file.name.split('.')[-1]} named {input_file.name}. Write and execute the necessary Python code to:
                1. Load the model from the file using joblib.
                2. Load the test data from the input file as a dataframe.
                3. Create a SHAP explainer for the model.
                4. Calculate the SHAP values for the test data.
                5. If the model is a multi-class classifier, extract SHAP values only for the class with the highest predicted probability.
                6. Add the SHAP values of each feature  for each record as new column in the dataframe with their names as SHAP_ feature_name so we know which feature the SHAP belongs to. The value for each feature's shap will be added not the whole array.
                7. Save the modified dataframe with SHAP values to a CSV file named 'test_data_with_shap.csv'.
                8. Save the SHAP values to a JSON file named 'shap_values.json' with keys, for shap values and feature names containing corresponding values.
                9. The json file should have the format {{
                "shap_values": [
                [record_1_shap_value1,record_1_shap_value2,record_1_shap_value3],
                [record_2_shap_value1,record_2_shap_value2,record_2_shap_value3],
                ...
            ],
            "feature_names": ["f1","f2"...]
        }}
            """
    shap_values_result = python_executor_wrapper(shap_values_prompt)

    if "Error" in shap_values_result:
        return 'Error'

    # Extract file paths from the result
    # output_files = json.loads(shap_values_result["output"])
    # csv_file = output_files["csv_file"]
    # json_file = output_files["json_file"]

    return shap_values_result

def generate_descriptions(shap_json, business_context):
    # Step 2: Generate Human-Readable Descriptions
    analysis_template = """
        You are given a JSON object containing SHAP values for a model. Here is the JSON object:

        {shap_json}

        which also contains the feature names for which shap values belong. For each record, generate a human-readable description explaining the prediction based on the SHAP values and feature values based on the business context {business_context}. Describe why the model predicted such a result for each record using the business context like the way people in that business can understand easily without any Data science background. Additionally, provide an overall summary highlighting any trends and features that have significant positive or negative impacts across all records.

        Example of a record-level description:
        "Record 1: The age feature (value: 65) contributed positively to the prediction of diabetes, indicating a higher likelihood due to the higher age. The physical activity feature (value: low) contributed negatively, indicating a reduced likelihood of diabetes due to higher physical activity."

        The overall summary should include:
        - Features with consistently high positive impact.
        - Features with consistently high negative impact.
        - Any emerging trends from the data.

        The output should be in the following JSON format:
        {{
            "individual_descriptions": [
                {{"index": 0, "description": "Record 1: ..."}},
                {{"index": 1, "description": "Record 2: ..."}},
                ...
            ],
            "overall_summary": "..."
        }}
        """
    print(type(analysis_template))
    shap_analysis_prompt = PromptTemplate(template = analysis_template, input_variables = ['shap_json','business_context'])
    llm = ChatOpenAI(temperature=0.5)
    chain = LLMChain(llm = llm, prompt = shap_analysis_prompt)
    shap_analysis_result = chain.invoke(input = {'shap_json':shap_json,'business_context':business_context})
    print(shap_analysis_result)
    if "Error" in shap_analysis_result:
        return shap_analysis_result

    text = shap_analysis_result.get('text')

    # Extract the JSON part of the response
    descriptions_json = json.loads(text)
    individual_descriptions = {desc["index"]: desc["description"] for desc in descriptions_json['individual_descriptions']}
    overall_summary = descriptions_json['overall_summary']

    # Load the original CSV file with SHAP values
    test_data_with_shap = pd.read_csv("test_data_with_shap.csv")

    # Add descriptions to the DataFrame
    test_data_with_shap['Description'] = test_data_with_shap.index.map(individual_descriptions)

    # Save the updated DataFrame to a new CSV file
    updated_csv_file = "record_descriptions_with_shap.csv"
    test_data_with_shap.to_csv(updated_csv_file, index=False)

    return overall_summary



# def load_and_explain(input_file, file, model):
#     # Initialize the LLM agent with Python REPL capabilities
#     instructions = """
#         You are an agent designed to write and execute python code to answer questions.
#         You have access to a python REPL, which you can use to execute python code. If you get an error, debug your code and try again.
#         Only use the output of your code to answer the question. You might know the answer without running any code, but you should still run the code to get the answer.
#         If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
#     """
#
#     base_prompt_py = hub.pull("langchain-ai/react-agent-template")
#     python_prompt = base_prompt_py.partial(instructions=instructions)
#     python_tools = [PythonREPLTool()]
#     python_agent = create_react_agent(
#         prompt=python_prompt,
#         llm=ChatOpenAI(temperature=0.2, model="gpt-4-turbo"),
#         tools=python_tools
#     )
#
#     python_agent_executor = AgentExecutor(agent=python_agent, tools=python_tools, verbose=True)
#
#     def python_executor_wrapper(original_prompt: str) -> dict[str, Any]:
#         return python_agent_executor.invoke({"input": original_prompt})
#
#     # Step 1: Generate SHAP Values and Save to CSV
#     shap_values_prompt = f"""
#             You are given a {type(model)} model, which you can load from the file named {file} using joblib. You also have an input file containing test data of type {input_file.name.split('.')[-1]} named {input_file.name}. Write and execute the necessary Python code to:
#             1. Load the model from the file using joblib.
#             2. Load the test data from the input file as a dataframe.
#             3. Create a SHAP explainer for the model using feature_perturbation='interventional'.
#             4. Calculate the SHAP values for the test data with check_additivity=False.
#             5. If the model is a multi-class classifier, extract SHAP values only for the class with the highest predicted probability.
#             6. Add the SHAP values for each record as new columns in the dataframe.
#             7. Save the modified dataframe with SHAP values to a CSV file named 'test_data_with_shap.csv'.
#             8. Save the SHAP values to a JSON file named 'shap_values.json'.
#         """
#     shap_values_result = python_executor_wrapper(shap_values_prompt)
#
#
#     if "Error" in shap_values_result:
#         return shap_values_result
#
#     # Extract file paths from the result
#     output_files = json.loads(shap_values_result["output"])
#     csv_file = output_files["csv_file"]
#     json_file = output_files["json_file"]
#
#     return {"csv_file": csv_file, "json_file": json_file}
#
# def generate_descriptions(shap_json, business_context):
#     # Step 2: Generate Human-Readable Descriptions
#     shap_analysis_prompt = PromptTemplate(f"""
#         You are given a JSON object containing SHAP values for a model. Here is the JSON object:
#
#         {json.dumps(shap_json)}
#
#         The features are {shap_json['features']}. For each record, generate a human-readable description explaining the prediction based on the SHAP values and feature values based on the business context {business_context}. Describe why the model predicted such a result for each record. Additionally, provide an overall summary highlighting any trends and features that have significant positive or negative impacts across all records.
#
#         Example of a record-level description:
#         "Record 1: The age feature (value: 65) contributed positively to the prediction of diabetes, indicating a higher likelihood due to the higher age. The physical activity feature (value: low) contributed negatively, indicating a reduced likelihood of diabetes due to higher physical activity."
#
#         The overall summary should include:
#         - Features with consistently high positive impact.
#         - Features with consistently high negative impact.
#         - Any emerging trends from the data.
#
#         Return the descriptions in a JSON format with individual descriptions and an overall summary.
#         """)
#     llm = ChatOpenAI(temperature=0.5,model='gpt-4-turbo')
#     shap_analysis_result = llm.invoke(shap_analysis_prompt)
#
#     if "Error" in shap_analysis_result:
#         return shap_analysis_result
#
#     descriptions_json = json.loads(shap_analysis_result["output"])
#     individual_descriptions = descriptions_json['individual_descriptions']
#     overall_summary = descriptions_json['overall_summary']
#
#     # Load the original CSV file with SHAP values
#     test_data_with_shap = pd.read_csv(shap_json["csv_file"])
#
#     # Add descriptions to the DataFrame
#     test_data_with_shap['Description'] = individual_descriptions
#
#     # Save the updated DataFrame to a new CSV file
#     updated_csv_file = "record_descriptions_with_shap.csv"
#     test_data_with_shap.to_csv(updated_csv_file, index=False)
#
#     return overall_summary

def process_input_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    else:
        st.error("Unsupported file type.")
        return None
#
#
# def predict(model, data):
#     predictions = model.predict(data)
#     return predictions
#
#
# def explain_predictions(model, data, predictions):
#     explanations = []
#     for i, prediction in enumerate(predictions):
#         explanation = f"Record {i + 1}: Feature1 contributed X, Feature2 contributed Y, etc."
#         explanations.append(explanation)
#     return explanations


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
