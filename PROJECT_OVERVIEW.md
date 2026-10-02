# DineSense AI: Project Overview & Technical Guide

## 1. What the Project Is About

Most customer feedback on review platforms like Zomato or Yelp is complex and nuanced. A customer rarely leaves a purely one-sided review; instead, they often say things like:
"The mutton biryani was outstanding, but our waiter took 45 minutes to bring the bill and the place was far too noisy."

Conventional sentiment analysis algorithms collapse this entire review into a single, generic label like "Positive" or "Neutral" based on the overall rating. This aggregate approach is unhelpful for restaurant operators because it fails to answer:
- Which specific part of the business satisfied the customer?
- Which specific department failed and needs operational intervention?

DineSense AI solves this problem by building an end-to-end Aspect-Based Sentiment Analysis (ABSA) and Operational Decision Intelligence platform. It breaks down raw reviews into individual clauses, evaluates sentiment across four distinct operational dimensions (Food, Service, Price, and Ambience), tracks the restaurant's operational health over time using Bayesian statistics, extracts exact root-cause complaints, and generates verified 30-day action plans for restaurant management.

---

## 2. What We Did

### Data Preprocessing & Leakage Prevention
- Cleaned and normalized a real-world dataset of 9,600+ restaurant reviews across 100 restaurants.
- Fixed data anomalies, stripped corrupted numeric columns, handled missing reviewer metadata, and removed duplicate reviews.
- Mapped raw 1 to 5 star ratings into three sentiment classes: Negative (ratings 1 to 2), Neutral (rating 3), and Positive (ratings 4 to 5).
- Designed grouped stratified splits (using GroupShuffleSplit and StratifiedGroupKFold) grouped by Reviewer ID. This ensures that no individual reviewer's writing style or bias leaks from the training set into the validation or test sets.

### Model Benchmarking & Fine-Tuning
- Implemented and evaluated classical machine learning baselines:
  - Majority Class baseline
  - TF-IDF with Logistic Regression
  - Linear Support Vector Classifier (Linear SVC)
  - Random Forest Classifier
  - XGBoost Classifier
- Fine-tuned deep transformer models using PyTorch, AdamW optimizer, and linear warmup schedules:
  - DistilBERT (distilbert-base-uncased, 66 million parameters): Selected for production deployment because it achieves 87.5% test accuracy and 0.747 Macro-F1 with low CPU inference latency (15 to 25 milliseconds per review), eliminating the need for expensive GPU hosting.
  - DeBERTa-v3 (microsoft/deberta-v3-base): Fine-tuned on GPU as a high-capacity research benchmark. Its disentangled relative attention mechanism handles complex linguistic phrasing and minority classes (lifting Neutral F1 to 0.510).

### Statistical Validation & Ground-Truth Benchmarking
- Validated performance differences using McNemar's test, confirming that DistilBERT's improvement over classical baselines is statistically significant (p < 1e-5).
- Measured model confidence reliability using Expected Calibration Error (ECE = 0.0164 / 1.6%), proving predicted probabilities accurately reflect true accuracy.
- Computed 95% Bootstrap Confidence Intervals over 1,000 resamples for Macro-F1 [0.716, 0.776].
- Created a 500-clause hand-annotated Gold Benchmark Dataset stratified across ratings and aspects, and verified annotation reliability using Cohen's Kappa (kappa = 0.812, representing substantial human agreement).

### Discourse Clause Segmentation & Aspect Matching
- Solved mixed-polarity sentence confusion using Rhetorical Structure Theory (RST). Concessive and contrasting conjunctions (like "despite", "although", "but", "however") are used to isolate contrasting clauses into independent semantic units.
- Built a two-tier hybrid aspect matcher:
  - Tier 1: Fast O(1) keyword lexicon search for direct aspect terms.
  - Tier 2: Dense semantic embedding fallback using sentence-transformers (all-MiniLM-L6-v2) to measure cosine similarity against precomputed aspect centroids, accurately classifying implicit phrases (such as "cost an arm and a leg" into Price).

### Root-Cause Complaint Extraction
- Applied SpaCy neural dependency parsing to extract grammatical modifier pairs directly from negative clauses:
  - Subject-Adjective pairs (e.g., "food was cold" -> target: food, opinion: cold)
  - Adjectival modifiers (e.g., "rude waiter" -> target: waiter, opinion: rude)
  - Negations (e.g., "not fresh" -> target: dish, opinion: not fresh)
- Clustered these grammatical pairs into top operational issues accompanied by verbatim customer quotes for evidence.

### Time-Decayed Empirical Bayes Health Scorecard
- Addressed the limitation of static average ratings, which penalize restaurants for old mistakes or inflate scores for low-volume venues.
- Formulated an exponential time-decay weighting (w = exp(-lambda * delta_t)) that assigns higher importance to recent customer experiences while discounting older feedback.
- Applied Empirical Bayes smoothing to pull low-sample restaurant scores toward the global category mean, preventing small-sample score distortion and yielding a reliable 0 to 100 operational health score.

### Grounded Operational Playbooks & Fact Verification
- Built an operational restaurant Standard Operating Procedure (SOP) repository that matches identified complaint clusters with targeted hospitality interventions (such as kitchen heat lamps, mobile POS order terminals, acoustic baffles, and menu re-engineering).
- Implemented regex fact-checking guardrails to verify that all figures, complaint frequencies, and percentages in generated summary reports match source data with 100% accuracy.

### Interactive Web Platform
- Built and launched an interactive Streamlit application featuring:
  - Restaurant Health Scorecards with aspect breakdowns (Food, Service, Price, Ambience).
  - Root-Cause Mining view with complaint clusters and 30-day action roadmaps.
  - Batch review analysis via CSV upload.
  - Live review sandbox testing for real-time clause extraction and sentiment scoring.
  - Model benchmarking dashboard comparing ML and transformer metrics.

---

## 3. Frameworks and Libraries Used

### Deep Learning & Transformer Architectures
- PyTorch: Deep learning tensor computations, training loops, backpropagation, and model checkpointing.
- Hugging Face Transformers: Loading pretrained architectures (DistilBERT, DeBERTa-v3), tokenizers (AutoTokenizer), model heads (AutoModelForSequenceClassification), AdamW optimization, and learning rate schedulers.
- Sentence-Transformers: Generating dense semantic sentence embeddings (all-MiniLM-L6-v2) for aspect matching.

### Natural Language Processing (NLP)
- SpaCy: Neural dependency parsing (nsubj, acomp, amod, neg tags), part-of-speech tagging, and lemmatization for grammatical root-cause extraction.
- NLTK: Tokenization, stopword filtering, and discourse conjunction patterns.

### Machine Learning, Statistics & Mathematics
- Scikit-learn: Building classical baselines (Logistic Regression, Linear SVC, Random Forest), TF-IDF vectorization, evaluation metrics (Accuracy, Macro-F1, Precision, Recall, Confusion Matrices), GroupShuffleSplit, and Cohen's Kappa score.
- XGBoost: Gradient-boosted decision tree baseline modeling.
- SciPy & NumPy: Numerical array operations, continuous exponential decay formulas, Bayesian shrinkage calculations, and bootstrap resampling.
- Pandas: Data cleaning, restructuring, aggregation, deduplication, and tabular data management.

### Web Deployment & Visualization
- Streamlit: Interactive production web dashboard, real-time inference widgets, data filtering, and scorecard rendering.
- Matplotlib & Seaborn: Plotting confusion matrices, calibration curves, and health index distributions.

---

## 4. Development Tools & Platforms Used

- Python (v3.10+): Primary programming language for all pipelines, training scripts, and web interfaces.
- Google Colab (NVIDIA T4 GPU): Cloud GPU compute environment used for fine-tuning the DeBERTa-v3 transformer backbone.
- Local CPU Environment: Optimized inference runtime for DistilBERT, SpaCy dependency parsing, and Streamlit execution.
- Visual Studio Code: Integrated development environment (IDE) for project development, debugging, and code organization.
- Git & GitHub: Version control, repository hosting, and code tracking.
