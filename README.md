# 🎓 Student Exam Performance Indicator

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **End-to-End Machine Learning Pipeline & Web Application** designed to analyze student demographic indicators, parental background, and test preparation factors to accurately predict student **Math Scores**.

---

## 📌 Project Overview

Understanding the factors that influence academic performance is crucial for educational institutions and parents. This project implements a complete modular machine learning lifecycle:
1. **Exploratory Data Analysis (EDA)** & feature engineering in Jupyter Notebooks.
2. **Automated Modular Pipeline** for Data Ingestion, Preprocessing/Transformation, and Model Training with Hyperparameter Tuning.
3. **Multi-Model Evaluation** testing 7 different regression algorithms to select the optimal model based on $R^2$ performance score.
4. **Interactive Web Application** built with Flask enabling real-time predictions via a simple web interface.

---

## 🏗️ Project Architecture & Directory Structure

```text
MachineLearningProject_pipeline/
├── artifacts/                  # Generated artifacts (data splits, fitted models)
│   ├── data.csv                # Raw processed dataset
│   ├── train.csv               # Training split
│   ├── test.csv                # Testing split
│   ├── preprocessor.pkl        # Fitted ColumnTransformer & Scaler object
│   └── model.pkl               # Best performing trained regression model
├── notebook/                   # Research & experimentation notebooks
│   ├── data/
│   │   └── stud.csv            # Original dataset
│   ├── EDA STUDENT PERFORMANCE.ipynb
│   └── MODEL TRAINING.ipynb
├── src/                        # Modular Python Package
│   ├── __init__.py
│   ├── components/             # Core Pipeline Components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py   # Data loading & train/test split
│   │   ├── data_transformation.py # Preprocessing, scaling & encoding
│   │   └── model_trainer.py    # Multi-model evaluation & tuning
│   ├── pipeline/               # Production Execution Pipelines
│   │   ├── __init__.py
│   │   ├── train_pipeline.py   # Training trigger pipeline
│   │   └── predict_pipeline.py # Model loader & prediction helper
│   ├── exception.py            # Custom exception handling with line-number tracing
│   ├── logger.py               # Automated timestamped logging module
│   └── utils.py                # Common utilities (object saving/loading, model eval)
├── templates/                  # Flask Web HTML Templates
│   ├── index.html              # Landing page
│   └── home.html               # Interactive prediction interface
├── app.py                      # Flask Application entry point
├── requirements.txt            # Project dependencies
├── setup.py                    # Packaging setup script
└── README.md                   # Project documentation
```

---

## 📊 Machine Learning Pipeline Workflow

### 1. Data Ingestion (`data_ingestion.py`)
- Reads the dataset (`stud.csv`) into a pandas DataFrame.
- Splits the dataset into **80% Training Set** and **20% Testing Set**.
- Exports `data.csv`, `train.csv`, and `test.csv` into the `artifacts/` folder.

### 2. Data Transformation (`data_transformation.py`)
- **Numerical Features** (`reading score`, `writing score`):
  - Handled missing values using `SimpleImputer(strategy='median')`.
  - Scaled features using `StandardScaler()`.
- **Categorical Features** (`gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course`):
  - Handled missing values using `SimpleImputer(strategy='most_frequent')`.
  - One-hot encoded features using `OneHotEncoder()`.
  - Scaled encoded features using `StandardScaler(with_mean=False)`.
- Combines preprocessing logic into a `ColumnTransformer` and saves it to `artifacts/preprocessor.pkl`.

### 3. Model Training & Evaluation (`model_trainer.py`)
Evaluates 7 Machine Learning algorithms using `GridSearchCV` for hyperparameter optimization:
- 🌲 **Random Forest Regressor**
- 🌳 **Decision Tree Regressor**
- 🚀 **Gradient Boosting Regressor**
- 📈 **Linear Regression**
- ⚡ **XGBoost Regressor (`XGBRegressor`)**
- 🐱 **CatBoost Regressor (`CatBoostRegressor`)**
- 🅰️ **AdaBoost Regressor**

The best performing model (yielding highest $R^2$ score) is automatically saved to `artifacts/model.pkl`.

---

## 🌐 Web Application (`app.py`)

The Flask web server provides an intuitive interface to input student details and obtain instant math score predictions.

### Input Features Required:
| Field Name | Type | Description / Allowed Values |
| :--- | :--- | :--- |
| **Gender** | Categorical | `Male`, `Female` |
| **Race/Ethnicity** | Categorical | `Group A` through `Group E` |
| **Parental Education** | Categorical | `bachelor's degree`, `some college`, `master's degree`, `associate's degree`, `high school`, `some high school` |
| **Lunch Type** | Categorical | `standard`, `free/reduced` |
| **Test Prep Course** | Categorical | `none`, `completed` |
| **Writing Score** | Numerical | `0` to `100` |
| **Reading Score** | Numerical | `0` to `100` |

### Output:
- **Predicted Math Score** (numerical score out of 100).

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+** installed on your system.
- Git installed on your system.

### 1. Clone or Open the Workspace
```bash
cd MachineLearningProject_pipeline
```

### 2. Create and Activate a Virtual Environment
```bash
# On Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies & Local Package
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 💻 Running the Application

### Step 1: Run Data Pipeline & Train the Model (Optional)
If you want to re-run data ingestion, preprocessing, and model training:
```bash
python src/components/data_ingestion.py
```
*This command will process the data, evaluate models, and regenerate `artifacts/preprocessor.pkl` and `artifacts/model.pkl`.*

### Step 2: Start the Flask Web App
```bash
python app.py
```

### Step 3: Access the Web App
Open your web browser and navigate to:
```text
http://127.0.0.1:5000/predictdata
```

---

## 📤 How to Push This Project to GitHub

Follow these steps to publish this repository to your GitHub account:

### Step 1: Initialize Git Repository
In your terminal / PowerShell prompt at the project root directory:
```bash
git init
```

### Step 2: Add Files & Make Initial Commit
```bash
git add .
git commit -m "Initial commit: End-to-End Machine Learning Student Performance Pipeline"
```

### Step 3: Create a New Repository on GitHub
1. Go to [GitHub New Repository Page](https://github.com/new).
2. Name your repository (e.g., `student-performance-ml-pipeline`).
3. Leave **"Initialize this repository with a README"** **UNCHECKED** (since we already created one).
4. Click **Create repository**.

### Step 4: Connect Remote & Push
Copy the commands shown on GitHub under **"…or push an existing repository from the command line"**:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/student-performance-ml-pipeline.git
git push -u origin main
```
*(Replace `YOUR_GITHUB_USERNAME` and repository name with your actual GitHub username and repository name).*

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).