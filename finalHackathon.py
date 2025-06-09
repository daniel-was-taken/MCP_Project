import gradio as gr
import pandas as pd
import io
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
import tempfile
import requests
import json
from openai import OpenAI # Added for Nebius AI Studio LLM integration

def load_data(file_input):
    """Loads CSV data from either a local file upload or a public URL."""
    if file_input is None:
        return None, None 

    try:
        if hasattr(file_input, 'name'):
            file_path = file_input.name
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif isinstance(file_input, str) and file_input.startswith('http'):
            response = requests.get(file_input)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
        else:
            return None, None

        # Extract column names here
        column_names = ", ".join(df.columns.tolist())
        return df, column_names
    except Exception as e:
        gr.Warning(f"Failed to load or parse data: {e}")
        return None, None


def update_detected_columns_display(file_data, url_data):
    """
    Detects and displays column names from the uploaded file or URL
    as soon as the input changes, before the main analysis button is pressed.
    """
    data_source = file_data if file_data is not None else url_data
    if data_source is None:
        return "" 

    df, column_names = load_data(data_source)
    if column_names:
        return column_names
    else:
        return "No columns detected or error loading file. Please check the file format."


def analyze_and_model(df, target_column):
    """Internal function to perform EDA, model training, and visualization."""
    profile = ProfileReport(df, title="EDA Report", minimal=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_html:
        profile.to_file(temp_html.name)
        profile_path = temp_html.name

    X = df.drop(columns=[target_column])
    y = df[target_column]
    task = "classification" if y.nunique() <= 10 else "regression"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LazyClassifier(ignore_warnings=True, verbose=0) if task == "classification" else LazyRegressor(ignore_warnings=True, verbose=0)
    models, _ = model.fit(X_train, X_test, y_train, y_test)

    sort_metric = "Accuracy" if task == "classification" else "R-Squared"
    best_model_name = models.sort_values(by=sort_metric, ascending=False).index[0] # Corrected indexing
    best_model = model.models[best_model_name]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_pkl:
        pickle.dump(best_model, temp_pkl)
        pickle_path = temp_pkl.name

    plt.figure(figsize=(10, 6))
    plot_column = "Accuracy" if task == "classification" else "R-Squared"
    sns.barplot(x=models[plot_column].head(10), y=models.head(10).index)
    plt.title(f"Top 10 Models by {plot_column}")
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_png:
        plt.savefig(temp_png.name)
        plot_path = temp_png.name
    plt.close()

    models_reset = models.reset_index().rename(columns={'index': 'Model'})
    return profile, profile_path, task, models_reset, plot_path, pickle_path

def run_pipeline(data_source, target_column, nebius_api_key):
    """
    This single function drives the entire application.
    It's exposed as the primary tool for the MCP server.

    :param data_source: A local file path (from gr.File) or a URL (from gr.Textbox).
    :param target_column: The name of the target column for prediction.
    :param nebius_api_key: The API key for Nebius AI Studio.
    """
    # --- 1. Input Validation ---
    if not data_source or not target_column:
        error_msg = "Error: Data source and target column must be provided."
        gr.Warning(error_msg)
        return None, error_msg, None, None, None, "Please provide all inputs.", "No columns loaded."

    gr.Info("Starting analysis...")

    # --- 2. Data Loading ---
    df, column_names = load_data(data_source) 
    if df is None:
        return None, "Error: Could not load data.", None, None, None, None, "No columns loaded."

    if target_column not in df.columns:
        error_msg = f"Error: Target column '{target_column}' not found in the dataset. Available: {column_names}"
        gr.Warning(error_msg)
        return None, error_msg, None, None, None, None, column_names 

    # --- 3. Analysis and Modeling ---
    profile, profile_path, task, models_df, plot_path, pickle_path = analyze_and_model(df, target_column)

    # --- 4. Explanation with Nebius AI Studio LLM ---
    best_model_name = models_df.iloc[0]['Model'] # Corrected indexing

    llm_explanation = "AI explanation is unavailable. Please provide a Nebius AI Studio API key to enable this feature." # Generic fallback [1]

    if nebius_api_key:
        try:
            client = OpenAI(
                base_url="https://api.studio.nebius.com/v1/",
            
                api_key=nebius_api_key
            )

            # Craft a prompt for the LLM [2]
            prompt_text = f"Explain and Summarize the significance of the top performing model, '{best_model_name}', for a {task} task in a data analysis context. Keep the explanation concise and professional. Analyse the report: {profile}."

            # Make the LLM call [2, 3]
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct", 
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that explains data science concepts. "},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.6, 
                max_tokens=512, 
                top_p=0.9,
                extra_body={
                    "top_k": 50
                } 
            )
            message_content = response.to_json()
            data = json.loads(message_content)
            llm_explanation = data['choices'][0]['message']['content']

        except Exception as e: 
            gr.Warning(f"Failed to get AI explanation: {e}. Please check your API key or try again later.")
            llm_explanation = "An error occurred while fetching AI explanation. Please check your API key or try again later."

    gr.Info("Analysis complete!")
    gr.Info(f'Profile report saved to: {profile_path}')
    return profile_path, task, models_df, plot_path, pickle_path, llm_explanation, column_names 

# --- Gradio UI ---
with gr.Blocks(title="AutoML Trainer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🤖 AutoML Trainer")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload Local CSV File")
            url_input = gr.Textbox(label="Or Enter Public CSV URL", placeholder="e.g., https://.../data.csv")
            target_column_input = gr.Textbox(label="Enter Target Column Name", placeholder="e.g., approved")
            nebius_api_key_input = gr.Textbox(label="Nebius AI Studio API Key (Optional)", type="password", placeholder="Enter your API key for AI explanations")
            run_button = gr.Button("Run Analysis & AutoML", variant="primary")

        with gr.Column(scale=2):
            column_names_output = gr.Textbox(label="Detected Columns", interactive=False, lines=2) # New Textbox for column names
            task_output = gr.Textbox(label="Detected Task", interactive=False)
            llm_output = gr.Markdown(label="AI Explanation")
            metrics_output = gr.Dataframe(label="Model Performance Metrics")

    with gr.Row():
        vis_output = gr.Image(label="Top Models Comparison")
        with gr.Column():
            eda_output = gr.File(label="Download Full EDA Report")
            model_output = gr.File(label="Download Best Model (.pkl)")

    def process_inputs(file_data, url_data, target, api_key):
        data_source = file_data if file_data is not None else url_data
        return run_pipeline(data_source, target, api_key)

    
    file_input.change(
        fn=update_detected_columns_display,
        inputs=[file_input, url_input],
        outputs=column_names_output
    )
    url_input.change(
        fn=update_detected_columns_display,
        inputs=[file_input, url_input],
        outputs=column_names_output
    )

    run_button.click(
        fn=process_inputs,
        inputs=[file_input, url_input, target_column_input, nebius_api_key_input],
        outputs=[eda_output, task_output, metrics_output, vis_output, model_output, llm_output, column_names_output] 
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_api=True,
    inbrowser=True,
    mcp_server=True
)