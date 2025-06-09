---
title: AutoML - MCP Hackathon
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: updated_ML.py
pinned: false
license: mit
tags:
  - machine-learning
  - mcp
  - hackathon
  - automl
  - lazypredict
  - gradio
  - mcp-server-track
  - agent-demo-track
short_description: Automated ML model comparison with LazyPredict and MCP integration
---

# 🤖 AutoML - MCP Hackathon Submission

**Automated Machine Learning Platform with LazyPredict and Model Context Protocol Integration**

## 🏆 Hackathon Track
**Agents & MCP Hackathon - Track 1: MCP Tool / Server**

## 🌟 Key Features

### Core ML Capabilities
- **📤 Dual Data Input**: Support for both local CSV file uploads and public URL data sources
- **🎯 Auto Problem Detection**: Automatically determines regression vs classification tasks
- **🤖 Multi-Algorithm Comparison**: LazyPredict-powered comparison of 20+ ML algorithms
- **📊 Automated EDA**: Comprehensive dataset profiling with ydata-profiling
- **💾 Best Model Export**: Download top-performing model as pickle file
- **📈 Performance Visualization**: Interactive charts showing model comparison results

### 🚀 Advanced Features
- **🌐 URL Data Loading**: Direct data loading from public CSV URLs with robust error handling
- **🔄 Agent-Friendly Interface**: Designed for both human users and AI agent interactions
- **📊 Interactive Dashboards**: Real-time model performance metrics and visualizations
- **🔍 Smart Error Handling**: Comprehensive validation and user feedback system
- **💻 MCP Server Integration**: Full Model Context Protocol server implementation

## 🛠️ How It Works

The AutoML provides a streamlined pipeline for automated machine learning:

### Core Functions

1. **`load_data(file_input)`** - Universal data loader that handles:
   - Local CSV file uploads through Gradio's file component
   - Public CSV URLs with HTTP/HTTPS support
   - Robust error handling and validation
   - Automatic format detection and parsing

2. **`analyze_and_model(df, target_column)`** - Core ML pipeline that:
   - Generates comprehensive EDA reports using ydata-profiling
   - Automatically detects task type (classification vs regression) based on target variable uniqueness
   - Trains and evaluates multiple models using LazyPredict
   - Selects the best performing model based on appropriate metrics
   - Creates publication-ready visualizations comparing model performance
   - Exports the best model as a serialized pickle file

3. **`run_pipeline(data_source, target_column)`** - Main orchestration function:
   - Validates all inputs and provides clear error messages
   - Coordinates the entire ML workflow from data loading to model export
   - Generates AI-powered explanations of results
   - Returns all outputs in a format optimized for both UI and API consumption

### Agent-Friendly Design
- **Single Entry Point**: The `run_pipeline()` function serves as the primary interface for AI agents
- **Flexible Input Handling**: Automatically determines whether input is a file path or URL
- **Comprehensive Output**: Returns all generated artifacts (models, reports, visualizations)
- **Error Resilience**: Robust error handling with informative feedback

## 🚀 Quick Start

### 📋 Application File Comparison

| Feature | `updated_ML.py` | `fixed_ML_MCP_backup.py` |
|---------|----------------|---------------------------|
| **Core ML Pipeline** | ✅ Full AutoML functionality | ✅ Full AutoML functionality |
| **MCP Server** | ✅ Enabled | ✅ Enhanced configuration |
| **UI Interface** | ✅ Clean, streamlined | ✅ Identical interface |
| **Code Structure** | ✅ Primary, well-documented | ✅ Backup with additional features |
| **Recommended For** | General use, development | Advanced MCP integration |

### Running the Application

The project includes two main application files:

#### Primary Application: `updated_ML.py` (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the main application
python updated_ML.py
```

#### Backup Version: `fixed_ML_MCP_backup.py`
```bash
# Alternative version with additional MCP features
python fixed_ML_MCP_backup.py
```

### Web Interface
1. **Choose Data Source**:
   - **Local Upload**: Use the file upload component to select a CSV file from your computer
   - **URL Input**: Enter a public CSV URL (e.g., from GitHub, data repositories, or cloud storage)
2. **Specify Target**: Enter the exact name of your target column (case-sensitive)
3. **Run Analysis**: Click "Run Analysis & AutoML" to start the AutoML pipeline
4. **Review Results**: 
   - View detected task type (classification/regression)
   - Examine model performance metrics in the interactive table
   - Download comprehensive EDA report (HTML format)
   - Download the best performing model (pickle format)
   - View model comparison visualization

### Installation & Setup
```bash
# Clone the repository
git clone [repository-url]
cd MCP_Project

# Install dependencies
pip install -r requirements.txt
```

### Server Configuration
The application launches with the following settings:
- **Host**: `0.0.0.0` (accessible from any network interface)
- **Port**: `7860` (default Gradio port)
- **MCP Server**: Enabled for AI agent integration
- **API Documentation**: Available at `/docs` endpoint
- **Browser Launch**: Automatic browser opening enabled

## 🎯 Current Implementation

### 1. LazyPredict Integration
- **Automated Model Training**: Trains 20+ algorithms automatically
- **Performance Comparison**: Side-by-side evaluation of all models
- **Best Model Selection**: Automatically selects top performer based on accuracy/R² score

### 2. Comprehensive EDA
- **ydata-profiling**: Generates detailed dataset analysis reports
- **Automatic Insights**: Data quality, distributions, correlations, and missing values
- **Interactive Reports**: Downloadable HTML reports with comprehensive statistics

### 3. Smart Task Detection
- **Classification**: Automatically detected when target has ≤10 unique values
- **Regression**: Automatically detected for continuous target variables
- **Adaptive Metrics**: Uses appropriate evaluation metrics for each task type

### 4. Model Persistence
- **Pickle Export**: Save trained models for future use
- **Model Reuse**: Load and apply models to new datasets
- **Production Ready**: Serialized models ready for deployment

## 📊 Supported Algorithms (via LazyPredict)

### Classification Algorithms
- Logistic Regression, Decision Tree Classifier
- Random Forest Classifier, Extra Trees Classifier
- Gradient Boosting Classifier, AdaBoost Classifier
- XGBoost Classifier, LightGBM Classifier
- SVM Classifier, K-Nearest Neighbors
- Naive Bayes, Linear Discriminant Analysis
- Quadratic Discriminant Analysis, and more...

### Regression Algorithms  
- Linear Regression, Ridge Regression, Lasso Regression
- Decision Tree Regressor, Random Forest Regressor
- Extra Trees Regressor, Gradient Boosting Regressor
- XGBoost Regressor, LightGBM Regressor
- Support Vector Regression, K-Nearest Neighbors
- AdaBoost Regressor, Elastic Net, and more...

## 🏆 Demo Scenarios

### House Price Prediction (Regression)
- Upload `sample_house_prices.csv` included in the project
- Enter `price` as the target column name
- System automatically detects regression task
- Compare performance of 15+ regression algorithms
- Download the best performing model and detailed EDA report

### Loan Approval Prediction (Classification)  
- Upload `sample_loan_approval.csv` included in the project
- Enter the loan approval status column name as target
- System automatically detects classification task
- Compare accuracy of 15+ classification algorithms
- Get comprehensive EDA report with approval insights

### College Placement Analysis
- Upload `collegePlace.csv` included in the project
- Analyze student placement outcomes
- Automatic feature analysis and model comparison
- Export trained model for future predictions

### URL-Based Data Analysis
- Use public dataset URLs for instant analysis
- Example: Government open data, research datasets, cloud-hosted files
- No file size limitations with URL-based loading
- Seamless integration with cloud storage platforms

## 🚀 Technologies Used

- **Frontend**: Gradio 4.0+ with soft theme and MCP server integration
- **AutoML Engine**: LazyPredict for automated model comparison and evaluation
- **EDA Framework**: ydata-profiling for comprehensive dataset analysis and reporting
- **ML Libraries**: scikit-learn, XGBoost, LightGBM (via LazyPredict ecosystem)
- **Visualization**: Matplotlib and Seaborn for model comparison charts and statistical plots
- **Data Processing**: pandas and numpy for efficient data manipulation and preprocessing
- **Model Persistence**: pickle for secure model serialization and export
- **Web Requests**: requests library for robust URL-based data loading
- **MCP Integration**: Model Context Protocol server for AI agent compatibility
- **File Handling**: tempfile for secure temporary file management

## 📈 Current Features

- **🔄 Dual Input Support**: Upload local CSV files or provide public URLs for data loading
- **🤖 One-Click AutoML**: Complete ML pipeline from data upload to trained model export
- **🎯 Intelligent Task Detection**: Automatic classification vs regression detection based on target variable analysis
- **📊 Multi-Algorithm Comparison**: Simultaneous comparison of 20+ algorithms with LazyPredict
- **📋 Comprehensive EDA**: Detailed dataset profiling with statistical analysis and data quality reports
- **💾 Model Export**: Download best performing model as pickle file for production deployment
- **📈 Performance Visualization**: Clear charts showing algorithm comparison and performance metrics
- **🌐 MCP Server Integration**: Full Model Context Protocol support for seamless AI assistant integration
- **🛡️ Robust Error Handling**: Comprehensive validation with informative user feedback
- **🎨 Modern UI**: Clean, responsive interface optimized for both human and agent interactions

## 🎯 Hackathon Submission Highlights

1. **🤖 LazyPredict Integration**: Automated comparison of 20+ ML algorithms with minimal configuration
2. **🧠 Smart Automation**: Intelligent task detection, data validation, and model selection
3. **📊 Comprehensive Analysis**: ydata-profiling powered EDA reports with statistical insights
4. **👥 Dual Interface Design**: Optimized for both human users and AI agent interactions
5. **🌐 MCP Server Implementation**: Full Model Context Protocol integration for seamless agent workflows
6. **🔄 Flexible Data Loading**: Support for both local uploads and URL-based data sources
7. **📈 Production Ready**: Exportable models, comprehensive documentation, and robust error handling
8. **🎨 Modern UI/UX**: Clean Gradio interface with intuitive workflow and clear feedback systems

## 📦 Project Structure

```
MCP_Project/
├── updated_ML.py             # Primary application file (recommended)
├── fixed_ML_MCP_backup.py    # Backup version with enhanced MCP features
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
├── uv.lock                  # UV dependency lockfile
├── README.md                # This documentation
├── sample_house_prices.csv  # Demo dataset for regression
├── sample_loan_approval.csv # Demo dataset for classification
├── collegePlace.csv         # Demo dataset for placement analysis
├── model_plot.png           # Sample visualization output
└── __pycache__/            # Python cache files
```

### Application Files Overview

- **`updated_ML.py`**: The main application file with clean, streamlined code structure. Recommended for most users.
- **`fixed_ML_MCP_backup.py`**: Alternative version with additional MCP server configurations and enhanced features.

Both files provide identical core functionality with slight variations in configuration and additional features.

## 📧 Contact & Support

Built with ❤️ for the **Agents & MCP Hackathon 2025**

This project demonstrates the power of combining LazyPredict's automated machine learning capabilities with the Model Context Protocol to create an intelligent, easy-to-use ML platform that seamlessly integrates into AI assistant workflows and provides production-ready machine learning solutions.

### 🔮 Features in Development
- 🧠 LLM-powered model explanations and insights
- ⚙️ Advanced feature engineering and preprocessing pipelines
- 🎯 Ensemble model creation and stacking capabilities
- 🚀 Real-time prediction API endpoints
- 🛠️ Enhanced MCP tool suite with additional ML operations
- 📊 Interactive model interpretation and SHAP value analysis

### 🎮 Usage Tips & Best Practices

#### Getting Started
- **Choose Your File**: Use `updated_ML.py` for standard usage, `fixed_ML_MCP_backup.py` for advanced MCP features
- **Target Column**: Ensure your target column name is exactly as it appears in the dataset (case-sensitive)
- **Data Sources**: Both local CSV uploads and public URLs are supported seamlessly

#### Data Loading Best Practices
- **URL Loading**: Use direct links to CSV files (GitHub raw URLs work great!)
- **File Size**: No strict limitations, but larger files may take longer to process
- **Data Quality**: The system handles missing values automatically, but clean data yields better results

#### Model Performance
- **Classification**: System uses Accuracy as the primary metric for model selection
- **Regression**: System uses R-Squared as the primary metric for model selection
- **File Formats**: Currently supports CSV format with automatic delimiter detection
- **Column Types**: Handles both numeric and categorical features automatically

#### Troubleshooting
- **Target Not Found**: Double-check column name spelling and case sensitivity
- **URL Issues**: Ensure URLs point directly to CSV files (not web pages)
- **Performance**: For large datasets, expect processing times of 2-5 minutes

---

**Ready to experience automated machine learning? Upload your dataset or provide a URL and let LazyPredict find the best algorithm for your problem!** 🚀

*Transform your data into insights with just a few clicks - no ML expertise required!*
