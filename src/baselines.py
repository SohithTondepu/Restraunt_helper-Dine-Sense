import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


def run_baselines():
    print("=== RUNNING CLASSICAL NLP BASELINES ON STRATIFIED SPLITS ===")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test.csv'))
    
    X_train = train_df['Review'].fillna('').tolist()
    y_train = train_df['Label'].tolist()
    
    X_val = val_df['Review'].fillna('').tolist()
    y_val = val_df['Label'].tolist()
    
    X_test = test_df['Review'].fillna('').tolist()
    y_test = test_df['Label'].tolist()
    
    print(f"Dataset sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # 1. Feature Extraction: TF-IDF with 1-2 grams
    print("Fitting TF-IDF Vectorizer (ngram_range=(1,2), max_features=10000)...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000,
        sublinear_tf=True
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_val)
    X_test_vec = tfidf.transform(X_test)
    
    # Save vectorizer
    joblib.dump(tfidf, os.path.join(MODELS_DIR, 'tfidf.pkl'))
    
    # 2. Define Model Configurations
    models = {
        'Majority Class Floor': DummyClassifier(strategy='most_frequent'),
        'TF-IDF + Logistic Regression': LogisticRegression(
            C=1.5,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'TF-IDF + Linear SVM': LinearSVC(
            C=0.8,
            class_weight='balanced',
            max_iter=2000,
            random_state=42
        ),
        'TF-IDF + Random Forest': RandomForestClassifier(
            n_estimators=150,
            class_weight='balanced',
            max_depth=30,
            random_state=42,
            n_jobs=-1
        ),
        'TF-IDF + XGBoost': XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )
    }
    
    results_list = []
    saved_models = {}
    label_names = ['Negative', 'Neutral', 'Positive']
    
    for name, model in models.items():
        print(f"\nTraining and evaluating: {name}...")
        model.fit(X_train_vec, y_train)
        saved_models[name] = model
        
        # Predict on Test set
        y_pred = model.predict(X_test_vec)
        
        acc = accuracy_score(y_test, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        p_class, r_class, f1_class, _ = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred)
        
        record = {
            'Model': name,
            'Architecture': 'Heuristic Floor' if 'Floor' in name else ('Linear ML' if 'Logistic' in name or 'SVM' in name else 'Tree Ensemble'),
            'Accuracy': round(acc * 100, 2),
            'Macro_Precision': round(p_macro, 4),
            'Macro_Recall': round(r_macro, 4),
            'Macro_F1': round(f1_macro, 4),
            'F1_Negative': round(f1_class[0], 4),
            'F1_Neutral': round(f1_class[1], 4),
            'F1_Positive': round(f1_class[2], 4),
            'Confusion_Matrix': cm.tolist()
        }
        results_list.append(record)
        print(f" -> Accuracy: {record['Accuracy']}% | Macro-F1: {record['Macro_F1']} | Neg-F1: {record['F1_Negative']} | Neu-F1: {record['F1_Neutral']} | Pos-F1: {record['F1_Positive']}")
        
    # Save serialized models
    joblib.dump(saved_models, os.path.join(MODELS_DIR, 'baseline_models.joblib'))
    
    # Save individual Logistic Regression for backward compatibility
    joblib.dump(saved_models['TF-IDF + Logistic Regression'], os.path.join(MODELS_DIR, 'LogisticRegression.joblib'))
    
    # Export results table
    results_df = pd.DataFrame(results_list)
    results_csv_path = os.path.join(RESULTS_DIR, 'sentiment_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    
    with open(os.path.join(RESULTS_DIR, 'baseline_metrics.json'), 'w') as f:
        json.dump(results_list, f, indent=2)
        
    print(f"\nBaseline results successfully written to {results_csv_path}")
    return results_df


if __name__ == '__main__':
    run_baselines()
