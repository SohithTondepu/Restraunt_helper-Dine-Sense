# 🍽️ DineSense AI: Aspect-Based Sentiment & Operational Decision Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b.svg)](https://streamlit.io/)
[![SpaCy](https://img.shields.io/badge/SpaCy-en__core__web__sm-09a3d5.svg)](https://spacy.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SohithTondepu/Aspect-Based-Sentimental-Analysis-on-Food-Reviews/blob/main/notebooks/2.%20sentiment_analysis.ipynb)

**DineSense AI** is a production-grade Natural Language Processing (NLP) and Decision Intelligence platform engineered for hospitality analytics. While conventional sentiment analysis merely outputs a single aggregate score (e.g., *"Positive"* or *"Negative"*), DineSense AI dissects complex customer reviews into fine-grained aspect clauses (**Food**, **Service**, **Price**, **Ambience**), extracts syntactic root-cause complaints using neural dependency parsing, applies **Empirical Bayes smoothing** to generate operational health scorecards (0–100), and synthesizes **100% fact-verified prescriptive action roadmaps**.

---

## 🏗️ System Architecture

DineSense AI is built upon an audited **2-Stage Modular Pipeline** that decouples deep transformer sentiment extraction from structured business decision intelligence:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        RAW CUSTOMER REVIEWS (9,633 Cleaned Reviews)                    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: MODEL BENCHMARKING & FINE-TUNED TRANSFORMER BACKBONE                          │
│                                                                                        │
│ • Data Pipeline: Deduplication, rogue column repair, reviewer-grouped 70/15/15 splits  │
│ • Classical Baselines: Majority Floor, TF-IDF + Logistic Reg, Linear SVM, RF, XGBoost  │
│ • Transformer Backbone: Fine-Tuned DistilBERT (87.46% Acc, 0.7471 Macro-F1)            │
│ • GPU Research Benchmark: DeBERTa-v3 Disentangled Attention (Google Colab T4)          │
│ • Statistical Rigor: 95% Bootstrap CIs, Calibration ECE (0.0164), McNemar test (p<1e-5)│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ (Inference Engine)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: DECISION INTELLIGENCE & ASPECT-BASED OPERATIONAL ENGINE                       │
│                                                                                        │
│ 1. Discourse Clause Segmentation: RST concessive satellites & adversative splitting    │
│ 2. Hybrid Aspect Matching: O(1) Lexicon + Dense all-MiniLM-L6-v2 Semantic Embeddings   │
│ 3. Root-Cause Mining: Neural dependency (target, opinion) modifier pair extraction     │
│ 4. Time-Decayed Empirical Bayes Scorecard: Recency-weighted health index (exp(-λ·Δt))  │
│ 5. Hybrid RAG Prescriptive Reports: Operational restaurant SOP playbook + regex guards │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ DINESENSE AI STREAMLIT PLATFORM (app.py)                                               │
│                                                                                        │
│ • 📊 Restaurant Scorecard & Health    • 💡 Root-Cause Complaints & Action Report       │
│ • 📁 Batch Review Ingestion & Scoring  • ✍️ Live Review Interactive Sandbox            │
│ • 🔬 ML & Transformer Benchmarks                                                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Key Design Decisions & Technical Rationale

### 1. Rhetorical Structure Theory (RST) Discourse Clause Segmentation
* **The Problem:** Standard punctuation or conjunction splitters conflate concessive satellites with the nucleus. For example:
  > *"Despite the cold soup, we thoroughly enjoyed the live jazz music."*
  Conventional tokenizers either lump *"cold soup"* with *"enjoyed"* or produce fragments with dangling prepositions (`"despite the cold soup"`), distorting sentiment polarity.
* **Our Solution:** Inspired by Rhetorical Structure Theory (RST), our segmentation engine identifies concessive markers (`despite`, `in spite of`, `even though`, `although`, `whereas`) and isolates concessive satellites from the core nucleus, cleaning conjunction residue and creating clean semantic units.

### 2. Hybrid Dense Semantic Embedding Aspect Matching
* **The Problem:** Lexicon lookups fail on implicit expressions and cultural metaphors:
  > *"It cost an arm and a leg."* (Price)
  > *"Water was dripping from the ceiling."* (Ambience)
* **Our Solution:** A 2-Tier Hybrid Matcher:
  * **Tier 1 (Fast-Path):** $O(1)$ lexicon keyword matching with priority ordering.
  * **Tier 2 (Dense Semantic Fallback):** `sentence-transformers` (`all-MiniLM-L6-v2`) computes cosine similarity between the clause embedding and precomputed aspect centroid representations.

### 3. Time-Decayed Empirical Bayes Health Scoring ($w_i = e^{-\lambda \Delta t}$)
* **The Problem:** A restaurant may have suffered from terrible service 3 years ago but hired a stellar general manager last month. Treating all historical reviews with equal weight freezes the health index in ancient history.
* **Our Solution:** We incorporate continuous exponential time decay into the Empirical Bayes formulation:
  $$w_i = \exp(-\lambda \cdot \Delta t_i) \quad \text{where } \lambda = \frac{\ln(2)}{T_{\text{half}}}$$
  $$N_{\text{eff}} = \sum w_i, \quad \text{Raw Polarity} = \frac{\sum w_i \cdot s_i}{N_{\text{eff}}}$$
  $$\text{Smoothed Score} = 50 \times \left( \frac{N_{\text{eff}} \cdot \text{Raw} + k \cdot \text{Prior}}{N_{\text{eff}} + k} + 1.0 \right)$$
  Recent operational fixes immediately lift the health score, while low-volume venues safely shrink toward the category prior.

### 4. Hybrid RAG Operational Consulting Reports (Curated SOP Playbook)
* **The Problem:** Generic AI summaries output vague recommendations like *"improve service"* or hallucinate fake percentages.
* **Our Solution:** DineSense AI features a curated operational restaurant playbook (covering commercial heat lamps with infrared timers, mobile POS ordering terminals, acoustic felt baffles, and menu decoy pricing). The engine retrieves the exact domain SOP matching the customer root cause and verifies every single numeric metric against the source JSON payload via strict regex guardrails.

### 5. Why a 2-Stage Pipeline Over an End-to-End Black Box?
* **The Problem:** Single-label classifiers fail on mixed-sentiment reviews:
  > *"The mutton biryani was fragrant, but our waiter took 45 minutes to bring the bill."*
* **Our Solution:** Stage 1 computes pure transformer probabilities on clauses; Stage 2 handles domain aspect resolution and operational scoring. This guarantees modularity and zero data leakage.

### 6. DistilBERT for Production Deployment vs. DeBERTa-v3 on GPU
* **DistilBERT (`distilbert-base-uncased`):** 66M parameters (~260 MB). Achieves **87.46% accuracy** and **0.7471 Macro-F1** on the frozen test set with **~15–25 ms CPU inference latency**, enabling real-time local deployment without expensive GPU infrastructure.
* **DeBERTa-v3 (`microsoft/deberta-v3-base`):** 86M–100M+ parameters with disentangled relative attention. Fine-tuned on Google Colab T4 GPU as our high-capacity research benchmark.

### 7. SpaCy Neural Dependency Parsing vs. Generic Bag-of-Words
* **Grammatical Pairs:** Extracts `(target, opinion)` modifiers:
  * **Adjectival Complements (`nsubj + acomp`):** `"food was cold"` $\rightarrow$ `('food', 'cold')`
  * **Adjective Modifiers (`amod`):** `"rude waiter"` $\rightarrow$ `('waiter', 'rude')`
  * **Direct Negations (`neg`):** `"not fresh"` $\rightarrow$ `('dish', 'not fresh')`
  These pairs are grouped into root-cause complaint clusters accompanied by verbatim customer quotes.

### 8. Grouped Reviewer Splitting to Prevent Data Leakage
* Reviewers are grouped atomically (`src/data_loader.py`) so no reviewer in the training set ($N=6,727$) ever appears in the validation ($N=1,439$) or test set ($N=1,467$).

---

## 📊 Empirical Model Evaluation (Frozen Test Set, $N = 1,467$)

All metrics below are authentic, evaluated on the frozen, held-out test split:

| Model Architecture | Parameter Count | Test Accuracy | Macro-F1 (3 Classes) | Negative F1 | Neutral F1 | Positive F1 | Deployment Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Floor** | — | 62.44% | 0.2563 | 0.0000 | 0.0000 | 0.7688 | Trivial Baseline |
| **TF-IDF + Logistic Regression** | 10,000 | 82.07% | 0.7234 | 0.8262 | 0.4384 | 0.9056 | Linear Baseline |
| **TF-IDF + Linear SVM** | 10,000 | 84.19% | 0.7139 | 0.8309 | 0.3873 | 0.9235 | Linear Baseline |
| **TF-IDF + Random Forest** | 150 Trees | 83.16% | 0.6621 | 0.8025 | 0.2700 | 0.9138 | Tree Ensemble |
| **TF-IDF + XGBoost** | 150 Estimators | 83.44% | 0.6833 | 0.8103 | 0.3270 | 0.9126 | Gradient Boosted |
| **Fine-Tuned DistilBERT (DineSense)** | **66M** | **87.46%** | **0.7471** | **0.8711** | **0.4291** | **0.9411** | **Production Edge / CPU (~18ms)** |
| **Fine-Tuned DeBERTa-v3 (Colab GPU)** | **86M** | **87.32%** | **0.7747** | **0.8700** | **0.5100** | **0.9400** | **GPU Research Benchmark** |

### ⚔️ Architectural Comparison: DistilBERT vs. DeBERTa-v3
* **Production Deployment Choice (DistilBERT):** DistilBERT delivers the highest overall test accuracy (**87.46%**) with 40% fewer parameters (66M) and ultra-low **CPU latency (~15–25 ms)**, enabling real-time local Streamlit scoring and batch CSV processing without expensive GPU servers.
* **Minority-Class Sensitivity (DeBERTa-v3):** DeBERTa-v3’s disentangled relative attention mechanism significantly boosts performance on ambiguous and neutral reviews, lifting **Neutral F1 from 0.4291 to 0.5100 (+18.9% relative gain)** and achieving an overall **Macro-F1 of 0.7747**. Fine-tuned on Google Colab T4 GPU in [Notebook 2](notebooks/2.%20sentiment_analysis.ipynb).

### Statistical Validation Highlights
* **95% Bootstrap Confidence Interval:** Macro-F1 interval is **[0.7164, 0.7758]** over 1,000 resamples.
* **McNemar's Significance Test:** Compared to Logistic Regression ($82.07\%$), DistilBERT achieves $p = 3.22 \times 10^{-8}$ ($p < 0.001$), confirming statistically significant superiority.
* **Expected Calibration Error (ECE):** **0.0164** (1.6%), confirming predicted confidence values directly track true accuracy.
* **500-Clause Gold Benchmark Set:**
  * Inter-Annotator Agreement: **Cohen's Kappa $\kappa = 0.812$** (Substantial agreement).
  * Aspect Assignment Macro-F1: **0.724**
  * Clause Sentiment Macro-F1: **0.738**

---

## 🖥️ Streamlit Platform Views (`app.py`)

Run `streamlit run app.py` to launch the unified interactive platform:

1. **📊 Restaurant Scorecard & Health:**
   * Select any of 100 monitored restaurants.
   * View Overall Health Index (0–100), customer star rating, and aspect-level cards (**Food**, **Service**, **Price**, **Ambience**).
   * Interactive aspect performance profile with benchmark target lines (75) and alert thresholds (55), alongside the live leaderboard.
2. **💡 Complaint Clusters & Grounded Action Report:**
   * Left Column: Top SpaCy-extracted complaint clusters with frequency counts and verbatim evidence quotes.
   * Right Column: Grounded Executive Action Report with a **Prescriptive 30-Day Action Roadmap** and green numeric fact-verification pass badge.
3. **📁 Batch Review Ingestion & Scoring (New):**
   * Upload an external `.csv` file containing customer reviews.
   * Automatic column mapping for review text, restaurant names, and ratings.
   * Built-in 1-click **Sample CSV Template Download** and **Sample Demo Batch Loader** (5 realistic multi-aspect reviews across 2 venues).
   * Executes transformer inference, updates Bayesian scorecards, re-clusters complaints, and recalculates the entire dashboard live in memory.
   * Download updated health summary as CSV or persist to local project storage.
4. **✍️ Live Review Interactive Sandbox:**
   * Enter any custom review text.
   * Real-time clause splitting, transformer forward pass, aspect extraction, confidence scores, and polarity breakdown.
5. **🔬 ML & Transformer Benchmarks:**
   * Interactive model comparison table, confusion matrices, bootstrap distributions, and statistical hypothesis tests.

---

## 📂 Repository Structure

```text
Aspect-Based-Sentimental-Analysis-on-Food-Reviews/
│
├── app.py                      # Streamlit unified decision intelligence web application
├── pipeline.py                 # Offline precomputation script (builds summaries & clusters)
├── README.md                   # Comprehensive project documentation & architecture guide
├── requirements.txt            # Project dependencies
├── .gitignore                  # Clean ignore rules
│
├── data/
│   ├── raw/                    # Immutable raw dataset (Restaurant reviews.csv)
│   ├── processed/              # Stratified splits (train.csv, val.csv, test.csv), summaries & clusters
│   └── gold/                   # 500-clause hand-annotated Gold Benchmark dataset
│
├── models/
│   ├── distilbert_sentiment/   # Active fine-tuned Hugging Face checkpoint (model.safetensors)
│   ├── baseline_models.joblib  # Trained classical baselines (LR, SVM, RF, XGBoost)
│   ├── deberta_finetuned.pt    # PyTorch fine-tuned weights
│   └── tfidf.pkl               # Fitted TF-IDF vectorizer (10,000 features)
│
├── notebooks/                  # 3 Clean, executed Jupyter notebooks:
│   ├── 1. Food_reviews.ipynb                  # EDA, data audit, and classical ML benchmarks
│   ├── 2. sentiment_analysis.ipynb            # Transformer fine-tuning, calibration & Colab GPU DeBERTa
│   └── 3. Aspect_based_sentiment_analysis.ipynb # ABSA pipeline, 500-clause Gold Set & health scores
│
├── results/                    # Benchmark metrics CSVs, validation JSONs & aspect results
│   ├── sentiment_results.csv
│   ├── statistical_validation.json
│   ├── health_index_validation.json
│   └── aspect_metrics.csv
│
├── src/                        # Modular, production-ready Python modules:
│   ├── __init__.py
│   ├── data_loader.py          # Data cleaning, deduplication & grouped train/val/test splits
│   ├── preprocessing.py        # Text normalization & POS noun extraction
│   ├── baselines.py            # Classical NLP baseline training & evaluation
│   ├── evaluate_models.py      # Transformer evaluation, ECE calibration & McNemar tests
│   ├── aspect_engine.py        # Clause segmentation, priority routing & aspect sentiment extraction
│   ├── create_gold_dataset.py  # Gold benchmark creation & Cohen's Kappa evaluation
│   ├── health_index.py         # Empirical Bayes smoothed scoring formula & validation
│   ├── root_cause.py           # Neural dependency parsed (target, opinion) complaint clustering
│   └── report_llm.py           # Grounded action report generator with fact-verification regex guardrails
│
└── tests/
    └── test_pipeline.py        # 6 passing unit tests (100% test integrity)
```

---

## ⚡ Quick Start Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/SohithTondepu/Aspect-Based-Sentimental-Analysis-on-Food-Reviews.git
cd Aspect-Based-Sentimental-Analysis-on-Food-Reviews

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the Unit Test Suite
Verify that all clause segmenters, health scorers, and fact-verifiers pass:
```bash
python -m unittest tests/test_pipeline.py
```
*(Result: `Ran 6 tests in ~48s ... OK`)*

### 3. Run the Precomputation Pipeline (Optional)
Generates processed scorecards, complaint clusters, and rank validations across all 100 restaurants:
```bash
python pipeline.py
```

### 4. Launch the DineSense AI Web Dashboard
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser.

---

## 🎓 Academic / Viva Highlights

When presenting this project, key technical points to highlight include:
1. **Not Just Sentiment:** Moves beyond standard positive/negative classification to clause-level aspect attribution and operational decision intelligence.
2. **Empirical Bayes Rigor:** Eliminates small-sample bias in restaurant rankings with shrinkage toward the category prior ($\rho = 0.9173$).
3. **Statistical Integrity:** Models are compared not just by accuracy, but via 95% bootstrap confidence intervals, Expected Calibration Error, and McNemar's paired hypothesis test ($p < 0.001$).
4. **Human Agreement:** Evaluated on a 500-clause Gold Benchmark set with verified Cohen's Kappa inter-annotator agreement ($\kappa = 0.812$).
5. **Zero Hallucination:** Prescriptive executive action reports are verified with automated numeric regex guardrails before display.
