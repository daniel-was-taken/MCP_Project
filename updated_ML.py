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

def load_data(file_input):
    """Loads CSV data from either a local file upload or a public URL."""
    if file_input is None:
        return None
    try:
        # For local file uploads, file_input is a temporary file object
        if hasattr(file_input, 'name'):
            file_path = file_input.name
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            df = pd.read_csv(io.BytesIO(file_bytes))
        # For URL text input
        elif isinstance(file_input, str) and file_input.startswith('http'):
            response = requests.get(file_input)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text))
        else:
            return None
        return df
    except Exception as e:
        gr.Warning(f"Failed to load or parse data: {e}")
        return None

def analyze_and_model(df, target_column):
    """Internal function to perform EDA, model training, and visualization."""
    # ... (This function's content is unchanged)
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
    best_model_name = models.sort_values(by=sort_metric, ascending=False).index[0]
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
    return profile_path, task, models_reset, plot_path, pickle_path

def run_pipeline(data_source, target_column):
    """
    This single function drives the entire application.
    It's exposed as the primary tool for the MCP server.
    
    :param data_source: A local file path (from gr.File) or a URL (from gr.Textbox).
    :param target_column: The name of the target column for prediction.
    """
    # --- 1. Input Validation ---
    if not data_source or not target_column:
        error_msg = "Error: Data source and target column must be provided."
        gr.Warning(error_msg)
        return None, error_msg, None, None, None, "Please provide all inputs."

    gr.Info("Starting analysis...")
    
    # --- 2. Data Loading ---
    df = load_data(data_source)
    if df is None:
        return None, "Error: Could not load data.", None, None, None, None

    if target_column not in df.columns:
        error_msg = f"Error: Target column '{target_column}' not found in the dataset. Available: {list(df.columns)}"
        gr.Warning(error_msg)
        return None, error_msg, None, None, None, None

    # --- 3. Analysis and Modeling ---
    profile_path, task, models_df, plot_path, pickle_path = analyze_and_model(df, target_column)
    
    # --- 4. Explanation ---
    best_model_name = models_df.iloc[0]['Model']
    llm_explanation = f"AI explanation for the '{task}' task: The top performing model was **{best_model_name}**."
    
    gr.Info("Analysis complete!")
    return profile_path, task, models_df, plot_path, pickle_path, llm_explanation

# --- Gradio UI ---
with gr.Blocks(title="Automated ML Playground", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🤖 Automated ML Playground")
    gr.Markdown("Enter a CSV data source (local file or public URL) and a target column to run the analysis. This interface is now friendly for both humans and AI agents.")

    with gr.Row():
        with gr.Column(scale=1):
            # Using gr.File allows for both upload and is compatible with agents
            file_input = gr.File(label="Upload Local CSV File")
            url_input = gr.Textbox(label="Or Enter Public CSV URL", placeholder="e.g., https://.../data.csv")
            target_column_input = gr.Textbox(label="Enter Target Column Name", placeholder="e.g., approved")
            run_button = gr.Button("Run Analysis & AutoML", variant="primary")
        
        with gr.Column(scale=2):
            task_output = gr.Textbox(label="Detected Task", interactive=False)
            llm_output = gr.Textbox(label="AI Explanation (WIP)", lines=3, interactive=False)
            metrics_output = gr.Dataframe(label="Model Performance Metrics")

    with gr.Row():
        vis_output = gr.Image(label="Top Models Comparison")
        with gr.Column():
            eda_output = gr.File(label="Download Full EDA Report")
            model_output = gr.File(label="Download Best Model (.pkl)")

    # The single click event that powers the whole app
    # A helper function decides whether to use the file or URL input
    def process_inputs(file_data, url_data, target):
        data_source = file_data if file_data is not None else url_data
        return run_pipeline(data_source, target)

    run_button.click(
        fn=process_inputs,
        inputs=[file_input, url_input, target_column_input],
        outputs=[eda_output, task_output, metrics_output, vis_output, model_output, llm_output]
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_api=True,
    inbrowser=True,
    mcp_server=True

)