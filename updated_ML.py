import gradio as gr
import pandas as pd
import io
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport

def load_data(file_bytes):
    if file_bytes is None:
        return None, gr.Dropdown(choices=['Target Column'],interactive=False)
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df, gr.Dropdown(choices=list(df.columns), interactive=True)

def analyze_and_model(df, target_column):
    # Generate EDA report
    profile = ProfileReport(df, title="EDA Report", minimal=True)
    profile_path = "eda_report.html"
    profile.to_file(profile_path)

    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    task = "classification" if y.nunique() <= 10 else "regression"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit models
    model = LazyClassifier() if task == "classification" else LazyRegressor()
    models, _ = model.fit(X_train, X_test, y_train, y_test)

    # Select best model based on accuracy (classification) or R2 (regression)
    sort_metric = "Accuracy" if task == "classification" else "R2 Score"
    best_model_name = models.sort_values(by=sort_metric, ascending=False).index[0]
    best_model = model.models[best_model_name]

    # Save the best model as pickle
    pickle_path = "best_model.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(best_model, f)

    # Visualization
    plt.figure(figsize=(10, 6))
    plot_column = "Accuracy" if task == "classification" else "R2 Score"
    sns.barplot(x=models[plot_column].head(10), y=models.head(10).index)
    plt.title(f"Top 10 Models by {plot_column}")
    plt.tight_layout()
    plot_path = "model_plot.png"
    plt.savefig(plot_path)
    plt.close()

    # Reset index to add model names as a column for Gradio display
    models_reset = models.reset_index().rename(columns={'index': 'Model'})

    return profile_path, task, models_reset, plot_path, pickle_path

def explain_with_llm(task, models_df):
    return "LLM explanation functionality is still a work in progress."

# Gradio UI
with gr.Blocks(title="automatedml_playground") as demo:
    gr.Markdown("## 🤖 Automated ML Playground")

    file_input = gr.File(label="Upload CSV", type="binary", file_types=[".csv"], file_count="single")
    column_dropdown = gr.Dropdown(label="Select Target Column", choices=['Target Column'], interactive=False)
    run_button = gr.Button("Run Analysis & AutoML", interactive=False)

    eda_output = gr.File(label="Download EDA Report")
    task_output = gr.Textbox(label="Detected Task", interactive=False)
    metrics_output = gr.Dataframe(label="Model Scores")
    llm_output = gr.Textbox(label="AI Explanation")
    vis_output = gr.Image(label="Top Models Comparison")
    model_output = gr.File(label="Download Best Model (.pkl)")

    df_state = gr.State()

    # Enable run_button when file is uploaded and column selected
    def enable_run_btn(df, col):
        return gr.Button(interactive=(df is not None and col is not None and col != ""))

    file_input.change(fn=load_data, inputs=file_input, outputs=[df_state, column_dropdown])
    column_dropdown.change(fn=enable_run_btn, inputs=[df_state, column_dropdown], outputs=run_button)

    run_button.click(
        fn=analyze_and_model,
        inputs=[df_state, column_dropdown],
        outputs=[eda_output, task_output, metrics_output, vis_output, model_output]
    )

    run_button.click(fn=explain_with_llm, inputs=[task_output, metrics_output], outputs=llm_output)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_api=True,
    inbrowser=True,
    mcp_server=True
)
