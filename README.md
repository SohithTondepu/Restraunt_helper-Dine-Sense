# 🍽️ Restaurant AI Decision Intelligence & ABSA Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DeBERTa-v3](https://img.shields.io/badge/Transformers-DeBERTa--v3-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/)

An enterprise-grade **2-Stage NLP & Decision Intelligence Platform** that converts customer review text into **fine-grained aspect sentiments** and **actionable operational business decisions** for 100+ restaurant brands.

---

## 🏗️ 2-Stage Architecture Overview

```
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1: ML & NLP MODEL BENCHMARKING (stage1_model_benchmarks.py)               │
 │                                                                                 │
 │ Classical ML (BoW / XGBoost)  vs.  Fine-Tuned BERT  vs.  DeBERTa-v3 (91.2% Acc) │
 └──────────────────────────────────────┬──────────────────────────────────────────┘
                                        │
                                        ▼ (High-Accuracy NLP Backbone)
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2: EXECUTIVE DECISION & BUSINESS INTELLIGENCE (app.py)                   │
 │                                                                                 │
 │ • Aspect Health Scores (0-100)       • Automated AI Operational Action Plans     │
 │ • Head-to-Head Competitive Radar     • Praise vs. Complaint Driver Extraction    │
 └─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 How to Run the Platforms

### 🔬 Stage 1: Run ML Model Benchmarking Dashboard
Inspect model training benchmarks, confusion matrices, failure modes, and live transformer inference:
```bash
streamlit run stage1_model_benchmarks.py
```
*Access at: `http://localhost:8501`*

---

### 🍽️ Stage 2: Run Executive Decision Intelligence Platform
Inspect restaurant aspect health scorecards, AI operational recommendations, and competitive benchmarks:
```bash
streamlit run app.py
```
*(Or `streamlit run stage2_decision_intelligence.py`)*

---

## 📂 Repository Structure

```
D:\Aspect-Based-Sentimental-Analysis-on-Food-Reviews
├── app.py                           # Stage 2 Main Streamlit Application
├── stage1_model_benchmarks.py       # Stage 1: ML Model Benchmarking Dashboard
├── stage2_decision_intelligence.py   # Stage 2: Executive Decision Intelligence App
├── requirements.txt                 # Dependencies
├── README.md                        # Project documentation
├── data/
│   └── Restaurant reviews.csv       # 10,000 reviews dataset across 100 restaurants
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             # POS Tagging & regex text cleaning
│   ├── aspect_engine.py             # Aspect extraction & sentiment engine
│   ├── decision_engine.py           # Stage 2 Health Scores & AI Recommendations
│   └── analytics.py                 # Benchmark & dataset analytics helpers
└── notebooks/                       # Research notebooks
    ├── 1. Food_reviews.ipynb
    ├── 2. sentiment_analysis.ipynb
    └── 3. Aspect_based_sentiment_analysis.ipynb
```

---

## 📊 Stage 1 Model Benchmarking Results

| Model | Model Family | Accuracy | F1-Score |
| :--- | :--- | :---: | :---: |
| **Logistic Regression** | Classical ML | 78.4% | 0.77 |
| **Decision Tree** | Classical ML | 71.2% | 0.70 |
| **Random Forest** | Classical ML | 81.5% | 0.81 |
| **XGBoost Classifier** | Classical ML | 83.1% | 0.83 |
| **BERT (`bert-base-uncased`)** | Fine-Tuned Transformer | 87.9% | 0.88 |
| **DeBERTa-v3 (`deberta-v3-base`)** | Fine-Tuned Transformer | **91.2%** | **0.91** |
