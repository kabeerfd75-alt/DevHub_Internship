#  Internship Projects — Kabeer Faqeer
**ID:** DHC-9122

---
# AI/ML Internship Tasks Portfolio

## Overview

This repository contains solutions for five AI/ML internship tasks covering machine learning pipelines, multimodal learning, retrieval-augmented generation (RAG), and large language model (LLM) applications. The projects demonstrate practical implementations of classification, regression, multimodal data fusion, conversational AI, and automated text tagging.

---

# Task 1: Customer Segmentation Using Machine Learning

## Objective

Build a machine learning solution to segment customers into meaningful groups based on behavioral and demographic data.

## Approach

* Data preprocessing and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Clustering using machine learning algorithms
* Customer segment visualization
* Performance evaluation using clustering metrics

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## Key Outcomes

* Identified customer groups with similar characteristics
* Improved customer targeting and personalization
* Generated actionable business insights

---

# Task 2: Customer Churn Prediction Pipeline

## Objective

Predict customer churn using a complete machine learning pipeline and automate preprocessing, training, and evaluation.

## Approach

* Data preprocessing
* Handling missing values
* Feature encoding
* Train-test split
* Model training and evaluation
* End-to-end ML pipeline implementation

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Pipeline API
* Machine Learning Classification Models

## Key Outcomes

* Automated prediction workflow
* Churn risk identification
* Reusable machine learning pipeline

---

# Task 3: Multimodal Housing Price Prediction

## Objective

Predict housing prices using both tabular property features and house images through a multimodal machine learning approach.

## Approach

* Combined structured housing data with image data
* Extracted visual features using Transfer Learning
* Used MobileNetV2 pretrained CNN as feature extractor
* Merged image embeddings with tabular features
* Trained regression model on combined features
* Evaluated performance using MAE and RMSE

## Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV2
* NumPy
* Pandas
* Scikit-learn

## Key Outcomes

* Demonstrated multimodal learning
* Leveraged pretrained CNNs without extensive training
* Improved property valuation using visual information

---

# Task 4: Context-Aware RAG Chatbot

## Objective

Develop a Retrieval-Augmented Generation (RAG) chatbot capable of answering questions using a custom knowledge base while maintaining conversation context.

## Approach

* Created document knowledge base
* Chunked text using RecursiveCharacterTextSplitter
* Generated embeddings using Sentence Transformers
* Stored embeddings in FAISS vector database
* Retrieved relevant documents through semantic search
* Maintained conversation history using memory modules
* Built interactive chatbot interface using Streamlit

## Technologies Used

* Python
* Streamlit
* LangChain
* FAISS
* HuggingFace Embeddings
* Sentence Transformers

## Key Outcomes

* Context-aware conversational AI
* Semantic document retrieval
* Memory-enabled chatbot interactions

---

# Task 5: Auto Tagging Support Tickets Using LLM

## Objective

Automatically classify customer support tickets and generate the top three most probable tags using LLM-inspired techniques.

## Approach

### Zero-Shot Classification

* Prompt-based ticket categorization
* Rule-based tagging without task-specific training

### Few-Shot Learning

* Provided example ticket-tag pairs
* Improved classification through contextual examples

### Fine-Tuned Alternative

* Implemented lightweight supervised text classifier
* TF-IDF feature extraction
* Logistic Regression model training
* Top-3 ranked tag prediction

## Technologies Used

* Python
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* NLP Techniques

## Key Outcomes

* Automated support ticket categorization
* Comparison of zero-shot and trained approaches
* Multi-class ranking with Top-3 predictions

---

# Project Structure

```bash
├── Task1/
├── Task2/
├── Task3/
├── Task4/
├── Task5/
├── datasets/
├── requirements.txt
└── README.md
```

---

# Skills Demonstrated

* Machine Learning
* Deep Learning
* Transfer Learning
* NLP
* LLM Applications
* Retrieval-Augmented Generation (RAG)
* Multimodal Learning
* Classification & Regression
* Feature Engineering
* Model Evaluation
* Prompt Engineering
* Vector Databases
* Conversational AI

---

# Author

**AI/ML Internship Project Portfolio**

This repository showcases practical implementations of modern Artificial Intelligence and Machine Learning techniques across customer analytics, predictive modeling, multimodal systems, retrieval-augmented generation, and large language model applications.
