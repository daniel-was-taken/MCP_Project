---
title: AutoML Playground - MCP Hackathon
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

# 🤖 AutoML Playground - MCP Hackathon Submission

**Automated Machine Learning Platform with LazyPredict and Model Context Protocol Integration**

## 🏆 Hackathon Track
**Agents & MCP Hackathon - Track 1: MCP Tool / Server**

## 🌟 Key Features

### Core ML Capabilities
- **📤 Smart CSV Upload**: Instant dataset loading and preprocessing
- **🎯 Auto Problem Detection**: Automatically determines regression vs classification
- **🤖 Multi-Algorithm Comparison**: LazyPredict-powered comparison of 20+ ML algorithms
- **📊 Automated EDA**: Comprehensive dataset profiling with ydata-profiling
- **💾 Best Model Export**: Download top-performing model as pickle file

### 🚀 Current Features
- **📈 Automated Model Comparison**: Compare multiple algorithms with one click
- **📊 Interactive Visualizations**: Top model performance charts
- **🔍 EDA Reports**: Comprehensive dataset analysis and profiling
- **💾 Model Persistence**: Save and download trained models
- **🎯 Smart Task Detection**: Automatic classification vs regression detection

### 🌐 MCP Integration
- **MCP Server Ready**: Model Context Protocol implementation
- **Gradio MCP Support**: Direct integration with AI assistants
- **API Endpoints**: RESTful interface for model training and predictions

## 🛠️ How It Works

The AutoML Playground uses LazyPredict to automatically train and compare multiple machine learning algorithms:

1. **`load_data()`** - Loads CSV files and extracts column names for target selection
2. **`analyze_and_model()`** - Core function that:
   - Generates comprehensive EDA reports using ydata-profiling
   - Automatically detects task type (classification vs regression)
   - Trains multiple models using LazyPredict
   - Selects the best performing model
   - Creates visualizations comparing model performance
   - Exports the best model as a pickle file
3. **`explain_with_llm()`** - Placeholder for future LLM-powered explanations

## 🚀 Quick Start

### Web Interface
1. **Upload** your CSV file using the file upload component
2. **Select** target column from the dropdown menu
3. **Click** "Run Analysis & AutoML" to start the automated process
4. **Review** the generated EDA report, model comparison results, and visualizations
5. **Download** the best performing model and EDA report

### Requirements
```bash
pip install -r requirements.txt
```

### Run the Application
```bash
python updated_ML.py
```

The application will launch on `http://localhost:7860` with:
- MCP server integration enabled
- API documentation available
- Browser auto-launch

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
- Select `price` as target column
- System automatically detects regression task
- Compare performance of 15+ regression algorithms
- Download the best performing model

### Loan Approval Prediction (Classification)
- Upload `sample_loan_approval.csv` included in the project
- Select loan approval status as target column
- System automatically detects classification task
- Compare accuracy of 15+ classification algorithms
- Get comprehensive EDA report with approval insights

### Custom Dataset Analysis
- Upload any CSV file with numeric/categorical data
- System automatically handles data types and missing values
- Get detailed EDA report with data quality insights
- Compare all relevant algorithms for your specific problem
- Export trained model for production use

## 🚀 Technologies Used

- **Frontend**: Gradio 4.0+ with MCP integration
- **AutoML**: LazyPredict for automated model comparison
- **EDA**: ydata-profiling for comprehensive dataset analysis
- **ML Libraries**: scikit-learn, XGBoost, LightGBM (via LazyPredict)
- **Visualizations**: Matplotlib, Seaborn for model comparison charts
- **Data Processing**: pandas, numpy for data manipulation
- **Model Persistence**: pickle for model serialization
- **MCP**: Model Context Protocol server integration

## 📈 Current Features

- **One-Click AutoML**: Upload CSV and get trained models instantly
- **Automatic Task Detection**: Smart classification vs regression detection
- **Multi-Algorithm Comparison**: Compare 20+ algorithms simultaneously
- **Comprehensive EDA**: Detailed dataset profiling and analysis
- **Model Export**: Download best performing model as pickle file
- **Performance Visualization**: Clear charts showing model comparison
- **MCP Integration**: Ready for AI assistant integration

## 🎯 Hackathon Submission Highlights

1. **LazyPredict Integration**: Automated comparison of 20+ ML algorithms
2. **Smart Automation**: Automatic task detection and model selection
3. **Comprehensive Analysis**: ydata-profiling powered EDA reports
4. **User-Friendly Interface**: Simple Gradio interface for non-technical users
5. **MCP Ready**: Model Context Protocol integration for AI assistants
6. **Production Ready**: Exportable models and API endpoints

## 📧 Contact & Support

Built with ❤️ for the **Agents & MCP Hackathon 2025**

This project demonstrates the power of combining LazyPredict's automated machine learning capabilities with the Model Context Protocol to create an intelligent, easy-to-use ML platform that can be seamlessly integrated into AI assistant workflows.

### Features in Development
- LLM-powered model explanations
- Advanced feature engineering
- Ensemble model creation
- Real-time prediction API
- Enhanced MCP tool suite

---

**Ready to experience automated machine learning? Upload your dataset and let LazyPredict find the best algorithm for your problem!** 🚀
