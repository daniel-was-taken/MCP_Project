"""
ML trainer MCP Server with Gradio Interface

This module provides a comprehensive machine learning model training interface
that can be used both as a standalone Gradio app and as an MCP server.
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import io
import base64
import json
import os
import warnings
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime
import tempfile

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
import warnings
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Additional ML libraries for enhanced features
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve, validation_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

class MLModelTrainer:
    def __init__(self):
        self.models = {
            
            'classification': {
                'Logistic Regression': LogisticRegression(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'Support Vector Machine': SVC(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB()
            }        
        }
        self.trained_models = {}

    def preprocess_data(self, df: pd.DataFrame, target_column: str,
                       handle_missing: str = 'drop', 
                       encode_categorical: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess the dataset"""
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Missing values before preprocessing: {X.isnull().sum().sum()}")
        
        # Identify categorical and numerical columns BEFORE any processing
        original_categorical_cols = X.select_dtypes(include=['object']).columns
        original_numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        print(f"Original categorical columns: {original_categorical_cols.tolist()}")
        print(f"Original numerical columns: {original_numerical_cols.tolist()}")
        
        # Handle missing values FIRST (before encoding)
        if handle_missing == 'drop':
            # Remove rows with missing values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            print(f"After dropping missing values: {X.shape}")
            
        elif handle_missing == 'mean':
            # Impute numerical columns with mean
            if len(original_numerical_cols) > 0:
                imputer_num = SimpleImputer(strategy='mean')
                X[original_numerical_cols] = imputer_num.fit_transform(X[original_numerical_cols])
            
            # Impute categorical columns with mode
            if len(original_categorical_cols) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                X[original_categorical_cols] = imputer_cat.fit_transform(X[original_categorical_cols])
        
        print(f"Missing values after handling: {X.isnull().sum().sum()}")
        
        # Handle categorical variables using ORIGINAL categorical columns
        if len(original_categorical_cols) > 0:
            print(f"Processing {len(original_categorical_cols)} categorical columns with {encode_categorical} encoding...")
            
            if encode_categorical == 'onehot':
                print("Using one-hot encoding...")
                X = pd.get_dummies(X, columns=original_categorical_cols, drop_first=True)
                print(f"After one-hot encoding: {X.shape}")

            elif encode_categorical == 'label':
                print("Using label encoding...")
                for col in original_categorical_cols:
                    print(f"Encoding column: {col}")
                    # Handle missing values specifically for this column if any remain
                    if X[col].isnull().any():
                        print(f"  Filling remaining NaN values in {col} with 'Unknown'")
                        X[col] = X[col].fillna('Unknown')
                    
                    # Convert to string and apply label encoding
                    X[col] = X[col].astype(str)
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    print(f"  {col}: {len(le.classes_)} unique values -> encoded to 0-{len(le.classes_)-1}")
          # Final validation - ensure all columns are numeric
        # Check for any remaining object columns that weren't properly encoded
        remaining_object_cols = X.select_dtypes(include=['object']).columns
        if len(remaining_object_cols) > 0:
            print(f"Warning: {len(remaining_object_cols)} columns still have object type after encoding: {remaining_object_cols.tolist()}")
            for col in remaining_object_cols:
                print(f"Processing remaining object column: {col}")
                # Try label encoding first for object columns
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    print(f"  Successfully label encoded {col}")
                except Exception as e:
                    print(f"  Failed to label encode {col}: {e}")
                    # As last resort, try numeric conversion
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                        print(f"  Converted {col} to numeric as fallback")
                    except Exception as e2:
                        print(f"  Failed to convert {col} to numeric: {e2}")
        
        # Final checks
        print(f"Final dataset shape: {X.shape}")
        print(f"Final missing values: {X.isnull().sum().sum()}")
        print(f"Data types: {X.dtypes.value_counts().to_dict()}")
          # Check for infinite values
        numerical_data = X.select_dtypes(include=[np.number])
        if len(numerical_data.columns) > 0:
            inf_count = np.isinf(numerical_data.to_numpy()).sum()
            print(f"Infinite values: {inf_count}")
            
            if inf_count > 0:
                print("Replacing infinite values with NaN and then with mean...")
                X[numerical_data.columns] = X[numerical_data.columns].replace([np.inf, -np.inf], np.nan)
                X[numerical_data.columns] = X[numerical_data.columns].fillna(X[numerical_data.columns].mean())
        print(f"Column dtypes: {X.dtypes.to_dict()}")
        print(f"Head of processed data:\n{X.head()}")
        return X, y 

    # def detect_problem_type(self, y: pd.Series) -> str:
    #     """Auto-detect if it's regression or classification"""
    #     if y.dtype in ['int64', 'float64']:
    #         unique_values = y.nunique()
    #         if unique_values <= 10 and y.dtype == 'int64':
    #             return 'classification'
    #         else:
    #             return 'regression'
    #     else:
    #         return 'classification'

    def train_model(self, file_path: str, target_column: str, 
                   problem_type: str, model_name: str,
                   test_size: float = 0.2, random_state: int = 42,
                   cross_validation: bool = True,
                   handle_missing: str = 'drop',
                   encode_categorical: str = 'label',
                   scale_features: bool = False) -> Dict[str, Any]:
        """Train a machine learning model"""
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                return {"error": "Only CSV files are supported"}
            
            if target_column not in df.columns:
                return {"error": f"Target column '{target_column}' not found in dataset"}
            
            # Auto-detect problem type if not specified
            # if problem_type == 'auto':
            #     problem_type = self.detect_problem_type(df[target_column])
            problem_type = "classification"


            # Preprocess data
            X, y = self.preprocess_data(df, target_column, handle_missing, encode_categorical)
            
            if len(X) == 0:
                return {"error": "No data remaining after preprocessing"}
            
            # Scale features if requested
            scaler = None
            if scale_features:
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Get model
            if model_name not in self.models[problem_type]:
                return {"error": f"Model '{model_name}' not available for {problem_type}"}
            
            model = self.models[problem_type][model_name]
          
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            if problem_type == 'regression':
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                metrics = {
                    'MSE': float(mse),
                    'MAE': float(mae),
                    'R²': float(r2),
                    'RMSE': float(np.sqrt(mse))
                }
            else:
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                metrics = {
                    'Accuracy': float(accuracy),
                    'Precision': float(report['weighted avg']['precision']),
                    'Recall': float(report['weighted avg']['recall']),
                    'F1-Score': float(report['weighted avg']['f1-score'])
                }
            
            # Cross-validation
            cv_scores = None
            if cross_validation:
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=5,
                    scoring='neg_mean_squared_error' if problem_type == 'regression' else 'accuracy'
                )
                cv_scores = cv_scores.tolist()
            
            # Store model and preprocessing objects
            model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.trained_models[model_id] = {
                'model': model,
                'scaler': scaler,
                'feature_names': X.columns.tolist(),
                'problem_type': problem_type,
                'model_name': model_name,
                'metrics': metrics,
            }
            
            preprocessed_df = pd.concat([X, pd.DataFrame(y)], axis=1)

            return {
                'success': True,
                'model_id': model_id,
                'problem_type': problem_type,
                'model_name': model_name,
                'metrics': metrics,
                'cv_scores': cv_scores,
                'feature_count': len(X.columns),
                'sample_count': len(X),
                'feature_names': X.columns.tolist(), 
                'preprocessed_df': preprocessed_df.head(10)
            }
            
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}


    def export_model(self, model_id: str) -> Optional[bytes]:
        """Export trained model as pickle"""
        if model_id not in self.trained_models:
            return None
        
        try:
            model_data = self.trained_models[model_id]
            
            # Create export package
            export_package = {
                'model': model_data['model'],
                'scaler': model_data['scaler'],
                'feature_names': model_data['feature_names'],
                'problem_type': model_data['problem_type'],
                'model_name': model_data['model_name'],
                'metrics': model_data['metrics'],
                'export_timestamp': datetime.now().isoformat()
            }
            
            return pickle.dumps(export_package)
            
        except Exception as e:
            print(f"Error exporting model: {e}")
            return None


    def predict_with_model(self, model_id: str, data: Dict) -> Dict[str, Any]:
        """Make predictions with a trained model"""
        if model_id not in self.trained_models:
            return {"error": "Model not found"}
        
        try:
            model_data = self.trained_models[model_id]
            model = model_data['model']
            scaler = model_data['scaler']
            feature_names = model_data['feature_names']
            
            # Convert data to DataFrame
            df = pd.DataFrame([data])
            
            # Ensure all features are present
            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0  # Default value for missing features
            
            # Reorder columns to match training data
            df = df[feature_names]
            
            # Scale if scaler was used
            if scaler:
                df = pd.DataFrame(scaler.transform(df), columns=feature_names)
            
            # Make prediction
            prediction = model.predict(df)[0]
            
            # Get prediction probability for classification
            prediction_proba = None
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(df)[0].tolist()
            
            return {
                'success': True,
                'prediction': float(prediction),
                'prediction_proba': prediction_proba,
                'model_name': model_data['model_name'],
                'problem_type': model_data['problem_type']
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

# Global instance
ml_trainer = MLModelTrainer()



def process_csv_upload(file_obj):
    """Process uploaded CSV file"""
    if file_obj is None:
        return None, "Please upload a CSV file", []
    
    try:
        df = pd.read_csv(file_obj.name)
        
        # Basic info
        info = f"""
        📊 **Dataset Info:**
        - Shape: {df.shape[0]} rows × {df.shape[1]} columns
        - Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB
        
        📋 **Columns:**
        {', '.join(df.columns.tolist())}
        
        🔍 **Data Types:**
        {df.dtypes.to_string()}
        
        📈 **Missing Values:**
        {df.isnull().sum().to_string()}
        """
        
        return df.head(10), info, list(df.columns)
        
    except Exception as e:
        return None, f"❌ Error reading CSV: {str(e)}", []

def train_model_interface(file_obj, target_column, problem_type, model_name, 
                         test_size, enable_tuning, enable_cv, handle_missing, 
                         encode_categorical, scale_features):
    """Gradio interface for model training"""
    if file_obj is None:
        return "❌ Please provide a CSV file or file path", None, None, None
    
    if not target_column:
        return "Please select a target column", None, None, None
    
    result = ml_trainer.train_model(
        file_path=file_obj.name,
        target_column=target_column,
        problem_type=problem_type,
        model_name=model_name,
        test_size=test_size,
        cross_validation=enable_cv,
        handle_missing=handle_missing,
        encode_categorical=encode_categorical,
        scale_features=scale_features
    )
    
    if 'error' in result:
        return f"❌ Training failed: {result['error']}", None, None, None
    
    # Format results
    metrics_text = "📊 **Performance Metrics:**  \n"
    for metric, value in result['metrics'].items():
        metrics_text += f"  * **{metric}: {value:.4f}**  \n"
    
    if result['cv_scores']:
        cv_mean = np.mean(result['cv_scores'])
        cv_std = np.std(result['cv_scores'])
        metrics_text += f"\n ### - 🔄 **Cross-Validation:**  "
        metrics_text += f"- Mean Score: {cv_mean:.4f} (±{cv_std:.4f})  "
    
    info_text = f""" 
    ## ✅ Model Training Successful! 
    ### - 🤖 **Model:** {result['model_name']}
    ### - 📝 **Problem Type:** {result['problem_type']}
    ### - 🆔 **Model ID:** {result['model_id']}
    ### - 📊 **Features:** {result['feature_count']}
    ### - 🎯 **Samples:** {result['sample_count']} 
    ### - {metrics_text}   
         
    """
    
    # Create download data
    model_bytes = ml_trainer.export_model(result['model_id'])
    if model_bytes:
        # Save to temporary file for download
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        temp_file.write(model_bytes)
        temp_file.close()
        download_path = temp_file.name
    else:
        download_path = None
    
    return info_text, download_path, result['preprocessed_df']  


# Create Gradio Interface
def create_gradio_app():
    """Create the main Gradio application"""
    
    with gr.Blocks(title="🤖 ML trainer", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🤖 ML trainer
        
        **A powerful machine learning platform that trains models on your CSV data and provides downloadable pickle files.**
        
        ### 🚀 Features:
        - **Auto Problem Detection**: Automatically detects regression vs classification
        - **Multiple Algorithms**: Support for 7 Classification ML algorithms
        - **Data Preprocessing**: Handle missing values, encode categories, scale features
        - **Hyperparameter Tuning**: Automated grid search optimization
        - **Model Evaluation**: Comprehensive metrics and visualizations
        - **Model Export**: Download trained models as pickle files
        - **MCP Server**: Use as a Model Context Protocol server
        """)
        
        with gr.Tab("📤 Train Model"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 Data Upload")
                      # MCP-compatible file input - handles both path strings and file objects
                    file_input = gr.File(
                        label="Upload CSV File",
                        file_types=[".csv"],
                        type="filepath"
                    )
                    
                    gr.Markdown("### ⚙️ Model Configuration")
                    target_column = gr.Dropdown(
                        label="Target Column",
                        choices=[],
                        interactive=True
                    )
                    
                    problem_type = gr.Radio(
                        label="Problem Type",
                        choices=["classification"],
                        value="classification"
                    )
                    
                    model_name = gr.Dropdown(
                        label="Model Algorithm",
                        choices=list(ml_trainer.models['classification'].keys()),
                        value="Random Forest"
                    )
                    
                    with gr.Accordion("🔧 Advanced Options", open=False):
                        test_size = gr.Slider(
                            label="Test Size",
                            minimum=0.1,
                            maximum=0.5,
                            value=0.2,
                            step=0.05
                        )
                        
                        enable_cv = gr.Checkbox(
                            label="Enable Cross-Validation",
                            value=True
                        )
                        
                        handle_missing = gr.Radio(
                            label="Handle Missing Values",
                            choices=["drop", "mean"],
                            value="drop"
                        )
                        
                        encode_categorical = gr.Radio(
                            label="Encode Categorical Variables",
                            choices=["onehot", "label"],
                            value="onehot"
                        )
                        
                        scale_features = gr.Checkbox(
                            label="Scale Features",
                            value=False
                        )
                    
                    train_btn = gr.Button("🚀 Train Model", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Data Preview")
                    data_preview = gr.Dataframe(
                        label="Dataset Preview",
                        interactive=False
                    )
                    
                    data_info = gr.Markdown()

            with gr.Column():
                gr.Markdown("### 📊 Dataset Preprocessed")
                data_processed_preview = gr.Dataframe(
                    label="Dataset Preprocessed Preview",
                    interactive=False
                )    
            
            
                
        with gr.Tab("📈 Model Evaluation"):  
            with gr.Row():
                training_results = gr.Markdown()
            
            with gr.Row():
                with gr.Column():
                    model_download = gr.File(
                        label="📥 Download Trained Model",
                        interactive=False
                    )

        
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ### 🎯 About ML trainer
            
            This application provides a comprehensive machine learning platform for training models on CSV datasets. 
            
            #### 🌟 Key Features:
            - **Automated Preprocessing**: Smart handling of missing values and categorical data
            - **Model Export**: Download trained models as pickle files
            - **MCP Integration**: Use as a Model Context Protocol server
            
            #### 🔧 Supported Algorithms:
            
            **Classification:**
            - Logistic Regression, Decision Tree
            - Random Forest, Gradient Boosting
            - SVM, KNN, Naive Bayes
            
            #### 📊 Evaluation Metrics:
            
            **Classification:** Accuracy, Precision, Recall, F1-Score
            
            #### 🚀 MCP Server Mode:
            
            This app can function as an MCP server, allowing integration with:
            - Claude Desktop
            - Cursor IDE
            - Any MCP-compatible client
            
            Built for the **Agents & MCP Hackathon Track 1** 🏆
            """)
        
        # Event handlers
        def update_target_column_choices(file_obj):
            """Update target column choices when file is uploaded"""
            df_preview, info, columns = process_csv_upload(file_obj)
            return df_preview, info, gr.Dropdown(choices=columns, value=None)
        
        # Handle file upload (primary method)
        file_input.upload(
            fn=update_target_column_choices,
            inputs=[file_input],
            outputs=[data_preview, data_info, target_column]
        )

        def update_model_choices(problem_type):
            return gr.Dropdown(choices=list(ml_trainer.models[problem_type].keys()))
        
        problem_type.change(
            fn=update_model_choices,
            inputs=[problem_type],
            outputs=[model_name]
        )
        
        train_btn.click(
            fn=train_model_interface,
            inputs=[
                file_input, target_column, problem_type, model_name,
                test_size, enable_cv, handle_missing,
                encode_categorical, scale_features
            ],
            outputs=[training_results, model_download, data_processed_preview]
        )
        
    return app

if __name__ == "__main__":
    # Create and launch the Gradio app
    app = create_gradio_app()
    # Launch the Gradio app
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=True,
        inbrowser=True, 
        mcp_server=True
    )






