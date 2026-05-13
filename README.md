#  Internship Projects — Kabeer Faqeer
**ID:** DHC-9122

---

## 📋 Table of Contents

- [Task 1 — Iris Dataset Exploration & Visualization](#task-1--iris-dataset-exploration--visualization)
- [Task 2 — Market Analysis](#task-2--market-analysis)
- [Task 3 — Heart Disease Prediction](#task-3--heart-disease-prediction)
- [Task 4 — AI-Based Health Assistant Chatbot](#task-4--ai-based-health-assistant-chatbot)
- [Task 5 — Mental Health Support Chatbot (Fine-Tuned)](#task-5--mental-health-support-chatbot-fine-tuned)
- [Task 6 — Predictive Model using Linear Regression](#task-6--predictive-model-using-linear-regression)

---

## Task 1 — Iris Dataset Exploration & Visualization

**File:** `KF_T1_15thMay`

### Objective
To perform Exploratory Data Analysis (EDA) on the classic Iris dataset to understand the relationships between different floral features.

### Key Technical Components

| Component | Description |
|-----------|-------------|
| **Data Loading** | Uses the `seaborn` library to load the built-in Iris dataset |
| **Data Inspection** | Checks dataset shape (150 rows, 5 columns), column names, and summary statistics using `.info()` and `.describe()` |
| **Visualizations** | Scatter plot comparing *Sepal Length* vs. *Sepal Width*, categorized by species to identify clusters |

### Libraries Used
`pandas` · `seaborn` · `matplotlib`

---

## Task 2 — Market Analysis

**File:** `KF_T2.ipynb`

### Objective
To fetch and process real-world financial data using the Yahoo Finance API for market analysis.

### Key Technical Components

| Component | Description |
|-----------|-------------|
| **Dependency Management** | Ensures installation of essential financial and data processing libraries like `yfinance` and `pandas` |
| **Environment Setup** | Configured for Google Colab with integrated data downloading capabilities |
| **Data Fetching** | Utilizes `curl_cffi` and `requests` for robust data retrieval from web sources |

### Libraries Used
`yfinance` · `pandas` · `numpy` · `requests` · `matplotlib` · `beautifulsoup4`

---

## Task 3 — Heart Disease Prediction

**File:** `KF_T3.ipynb`

### Objective
To build a machine learning model that predicts the likelihood of heart disease based on patient health records, using logistic regression for binary classification.

### Workflow
1. Import required libraries
2. Load dataset (`heart.csv`)
3. Perform null value checking
4. Visualize target distribution using countplot
5. Generate correlation heatmap
6. Split dataset into training and testing sets
7. Train logistic regression model
8. Predict outcomes
9. Evaluate using Accuracy, Confusion Matrix, and ROC Curve
10. Identify important features using model coefficients
11. Export processed dataset to CSV

### Evaluation Metrics
- ✅ Accuracy Score
- 📊 Confusion Matrix
- 📈 ROC Curve
- 🔍 Feature Importance (via model coefficients)

### Key Features
- Binary disease prediction
- Correlation analysis
- Performance visualization
- Feature ranking

### Libraries Used
`Python` · `pandas` · `seaborn` · `matplotlib` · `scikit-learn`

---

## Task 4 — AI-Based Health Assistant Chatbot

**File:** `KF_T4`

### Objective
To develop an intelligent healthcare chatbot that provides informational medical responses using state-of-the-art NLP models, while maintaining a history of conversations for further analysis.

### Core Components

#### 1. Backend & Model Logic — `KF_T4.ipynb`
- **Model Architecture:** Powered by `facebook/opt-350m` transformer model from Hugging Face for high-quality text generation
- **Conversational Pipeline:** Implements a dedicated chatbot function to process medical queries and generate context-aware responses
- **Data Management:** Logging system that captures user inputs and AI responses, exported to `chatbot_interactions.csv`

#### 2. Custom Web Interface — `task4_health_chatbot_improved.html`
- **Medical-Themed Design:** Clean, responsive layout using a professional blue and slate color palette suited for healthcare applications
- **Interactive Elements:** Smooth chat animations, typing indicators (3-dot bounce), and scrollable message bubbles
- **Typography:** Integrated *Inter* and *JetBrains Mono* fonts for enhanced readability and modern aesthetics

### Technology Stack
- **Languages:** Python · JavaScript · HTML5 · CSS3
- **AI Frameworks:** Hugging Face Transformers · PyTorch
- **Data Handling:** Pandas · NumPy
- **Environment:** Google Colab

### Workflow
1. Environment setup and installation of `transformers` and `accelerate`
2. Loading the pre-trained `facebook/opt-350m` model and tokenizer
3. Defining health assistant logic to handle user prompts and generate responses
4. Logging interaction data into a structured CSV format
5. Integration with the custom-built Web UI for a professional presentation

---

## Task 5 — Mental Health Support Chatbot (Fine-Tuned)

**File:** `KF_T5_Fixed_(1).ipynb`

### Objective
To develop a conversational agent capable of providing empathetic support for emotional wellness, stress, and anxiety using a fine-tuned Large Language Model.

### Model & Approach

| Component | Details |
|-----------|---------|
| **Model Base** | `microsoft/DialoGPT-small` — pre-trained for conversational response generation |
| **Fine-Tuning** | Hugging Face `Trainer` API with `DataCollatorForLanguageModeling` |
| **Dataset** | 50 custom empathetic conversation pairs covering 9 emotional categories |
| **Safety System** | Custom Crisis Keyword Detection with immediate support resource routing |

### Training Dataset — Emotional Categories Covered
- 😰 Anxiety & Sleep Issues
- 🤝 Loneliness & Isolation
- 😓 Overwhelm & Stress
- 😢 Sadness & Low Mood
- 💔 Self-Worth & Confidence
- 👥 Relationship Struggles
- 💼 Work & Life Pressure
- 💬 General Emotional Support
- 🚨 Crisis Detection & Safety Response

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch Size | 4 |
| Learning Rate | 5e-5 |
| Max Token Length | 128 |
| Optimizer | Adafactor |
| Warmup Steps | 10 |

### Safety Feature — Crisis Keyword Detection
The chatbot monitors user input for safety-critical terms (e.g., *"suicide"*, *"harm myself"*, *"want to die"*). Upon detection, it immediately returns a predefined crisis response with local and international helpline resources:
- 🇵🇰 **Pakistan:** Umang Helpline — `0317-4288665`
- 🌍 **International:** iCall — `icallhelpline.org`

### Interface & Deployment
- **Gradio UI:** Interactive, web-based chat interface with real-time empathetic response testing
- **Theme:** Soft blue/slate palette using `gr.themes.Soft` for a calming, accessible experience
- **Bot Greeting:** *"Hello! 💙 I'm your Mental Health Support companion. I'm here to listen without judgment. How are you feeling today?"*

### Response Generation Parameters

| Parameter | Value |
|-----------|-------|
| Max New Tokens | 100 |
| Temperature | 0.75 |
| Top-p | 0.92 |
| Top-k | 50 |
| Repetition Penalty | 1.3 |

### Technology Stack
`Python` · `PyTorch` · `Hugging Face Transformers` · `Datasets` · `Gradio` · `Pandas`

### Outputs
- Fine-tuned dialogue model saved to `./mental_health_bot_final`
- Live Gradio link for interactive testing
- Crisis-safe response system

---

## Task 6 — Predictive Model using Linear Regression

**File:** `KF_T_6.ipynb`

### Objective
To build and evaluate a predictive model using Linear Regression to estimate continuous numerical values.

### Key Technical Components

| Component | Description |
|-----------|-------------|
| **Preprocessing** | Splits data into training and testing sets using `train_test_split` for model generalizability |
| **Model Implementation** | Employs the `LinearRegression` algorithm from `scikit-learn` |
| **Evaluation Metrics** | Mean Absolute Error (MAE) and Mean Squared Error (MSE) to quantify prediction accuracy |

### Libraries Used
`scikit-learn` · `pandas` · `numpy` · `seaborn` · `matplotlib`

---

> 📌 *All tasks were completed as part of an internship program. Each project is self-contained with its own dataset, model, and evaluation pipeline.*
