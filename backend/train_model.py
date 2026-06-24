import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROCESSED_DIR = "C:/Users/shaisty priya/.gemini/antigravity-ide/scratch/ai-disease-prediction-system/dataset/processed"
MODELS_DIR = "C:/Users/shaisty priya/.gemini/antigravity-ide/scratch/ai-disease-prediction-system/models"

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Loading preprocessed datasets...")
    
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train_clean.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test_clean.csv"))
    
    # Split into features and target
    X_train = train_df.drop(columns=['prognosis'])
    y_train = train_df['prognosis']
    X_test = test_df.drop(columns=['prognosis'])
    y_test = test_df['prognosis']
    
    # Save the symptom features list
    symptom_list = list(X_train.columns)
    joblib.dump(symptom_list, os.path.join(MODELS_DIR, "symptom_encoder.pkl"))
    print(f"Saved list of {len(symptom_list)} symptoms to symptom_encoder.pkl")
    
    # Encode target labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Save disease encoder
    joblib.dump(le, os.path.join(MODELS_DIR, "disease_encoder.pkl"))
    print(f"Saved disease label encoder to disease_encoder.pkl")
    
    # Define models to compare
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Extra Trees": ExtraTreesClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss')
    }
    
    best_acc = 0.0
    best_model_name = ""
    best_base_model = None
    results = {}
    
    print("\nTraining and evaluating models on testing dataset...")
    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train_encoded)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test_encoded, y_pred)
        prec = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
        
        results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
        print(f"{name} Performance: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_base_model = clf
            
    print(f"\nBest Performing Model: {best_model_name} with {best_acc:.4f} accuracy.")
    
    # Hyperparameter Optimization on the best model type
    print(f"\nOptimizing hyperparameters for {best_model_name}...")
    if best_model_name == "Random Forest" or best_model_name == "Extra Trees":
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(
            estimator=best_base_model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring='accuracy'
        )
    elif best_model_name == "XGBoost":
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.1, 0.2]
        }
        grid_search = GridSearchCV(
            estimator=best_base_model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring='accuracy'
        )
    else:  # Gradient Boosting
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2]
        }
        grid_search = GridSearchCV(
            estimator=best_base_model,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            scoring='accuracy'
        )
        
    grid_search.fit(X_train, y_train_encoded)
    optimized_model = grid_search.best_estimator_
    
    # Re-evaluate optimized model
    y_pred_opt = optimized_model.predict(X_test)
    opt_acc = accuracy_score(y_test_encoded, y_pred_opt)
    print(f"Optimized {best_model_name} Params: {grid_search.best_params_}")
    print(f"Optimized {best_model_name} Accuracy: {opt_acc:.4f}")
    
    # Save the optimized model
    joblib.dump(optimized_model, os.path.join(MODELS_DIR, "disease_model.pkl"))
    print(f"\nSaved final model to disease_model.pkl")
    
    # Write a quick report file
    report_path = os.path.join(MODELS_DIR, "model_comparison_report.json")
    import json
    with open(report_path, 'w') as f:
        json.dump({
            "results": results,
            "best_model": best_model_name,
            "best_params": grid_search.best_params_,
            "final_accuracy": opt_acc
        }, f, indent=4)
    print(f"Saved comparison report to models/model_comparison_report.json")

if __name__ == "__main__":
    train()
