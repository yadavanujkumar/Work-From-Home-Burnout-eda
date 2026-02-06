"""
Work From Home Burnout Analysis - Supervised Learning Models
This module implements classification and regression models for burnout prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support,
                            mean_squared_error, r2_score, mean_absolute_error)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath='work_from_home_burnout_dataset.csv'):
    """Load and preprocess the dataset."""
    df = pd.read_csv(filepath)
    
    # Encode categorical variables
    le_day = LabelEncoder()
    df['day_type_encoded'] = le_day.fit_transform(df['day_type'])
    
    le_risk = LabelEncoder()
    df['burnout_risk_encoded'] = le_risk.fit_transform(df['burnout_risk'])
    
    return df, le_day, le_risk


def prepare_features(df):
    """Prepare feature matrix and target variables."""
    # Feature columns
    feature_cols = ['day_type_encoded', 'work_hours', 'screen_time_hours', 
                   'meetings_count', 'breaks_taken', 'after_hours_work', 
                   'sleep_hours', 'task_completion_rate']
    
    X = df[feature_cols]
    y_classification = df['burnout_risk_encoded']
    y_regression = df['burnout_score']
    
    return X, y_classification, y_regression, feature_cols


def classification_models(X, y, feature_names):
    """Train and evaluate classification models."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION MODELS - BURNOUT RISK PREDICTION")
    print("=" * 70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 1. Logistic Regression
    print("\n1. Logistic Regression")
    print("-" * 50)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr))
    
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'model': lr,
        'predictions': y_pred_lr
    }
    
    # 2. Random Forest Classifier
    print("\n2. Random Forest Classifier")
    print("-" * 50)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)
    
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'model': rf,
        'predictions': y_pred_rf,
        'feature_importance': feature_importance
    }
    
    # 3. XGBoost Classifier
    print("\n3. XGBoost Classifier")
    print("-" * 50)
    xgb_clf = xgb.XGBClassifier(n_estimators=100, random_state=42, 
                                eval_metric='mlogloss', use_label_encoder=False)
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_xgb))
    
    results['XGBoost'] = {
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'model': xgb_clf,
        'predictions': y_pred_xgb
    }
    
    # Visualization: Model comparison
    plot_classification_results(results, y_test)
    
    return results, X_test, y_test


def regression_models(X, y, feature_names):
    """Train and evaluate regression models."""
    print("\n" + "=" * 70)
    print("REGRESSION MODELS - BURNOUT SCORE PREDICTION")
    print("=" * 70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # 1. Linear Regression
    print("\n1. Linear Regression")
    print("-" * 50)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    r2_lr = r2_score(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    
    print(f"R² Score: {r2_lr:.4f}")
    print(f"RMSE: {rmse_lr:.4f}")
    print(f"MAE: {mae_lr:.4f}")
    
    results['Linear Regression'] = {
        'r2': r2_lr,
        'rmse': rmse_lr,
        'mae': mae_lr,
        'predictions': y_pred_lr
    }
    
    # 2. Random Forest Regressor
    print("\n2. Random Forest Regressor")
    print("-" * 50)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    
    print(f"R² Score: {r2_rf:.4f}")
    print(f"RMSE: {rmse_rf:.4f}")
    print(f"MAE: {mae_rf:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)
    
    results['Random Forest'] = {
        'r2': r2_rf,
        'rmse': rmse_rf,
        'mae': mae_rf,
        'predictions': y_pred_rf,
        'feature_importance': feature_importance
    }
    
    # 3. XGBoost Regressor
    print("\n3. XGBoost Regressor")
    print("-" * 50)
    xgb_reg = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_reg.fit(X_train, y_train)
    y_pred_xgb = xgb_reg.predict(X_test)
    
    r2_xgb = r2_score(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    
    print(f"R² Score: {r2_xgb:.4f}")
    print(f"RMSE: {rmse_xgb:.4f}")
    print(f"MAE: {mae_xgb:.4f}")
    
    results['XGBoost'] = {
        'r2': r2_xgb,
        'rmse': rmse_xgb,
        'mae': mae_xgb,
        'predictions': y_pred_xgb
    }
    
    # Visualization: Model comparison
    plot_regression_results(results, y_test)
    
    return results, X_test, y_test


def plot_classification_results(results, y_test):
    """Plot classification model results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    
    axes[0].bar(models, accuracies, color=['skyblue', 'coral', 'lightgreen'], edgecolor='black')
    axes[0].set_title('Classification Model Accuracy Comparison', fontweight='bold')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0, 1.1])
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Confusion matrix for best model
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    cm = confusion_matrix(y_test, results[best_model]['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title(f'Confusion Matrix - {best_model}', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: classification_results.png")


def plot_regression_results(results, y_test):
    """Plot regression model results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R² Score comparison
    models = list(results.keys())
    r2_scores = [results[m]['r2'] for m in models]
    
    axes[0, 0].bar(models, r2_scores, color=['skyblue', 'coral', 'lightgreen'], edgecolor='black')
    axes[0, 0].set_title('R² Score Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('R² Score')
    for i, v in enumerate(r2_scores):
        axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # RMSE comparison
    rmse_scores = [results[m]['rmse'] for m in models]
    axes[0, 1].bar(models, rmse_scores, color=['skyblue', 'coral', 'lightgreen'], edgecolor='black')
    axes[0, 1].set_title('RMSE Comparison (Lower is Better)', fontweight='bold')
    axes[0, 1].set_ylabel('RMSE')
    for i, v in enumerate(rmse_scores):
        axes[0, 1].text(i, v + 0.5, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Predictions vs Actual for best model
    best_model = max(results, key=lambda x: results[x]['r2'])
    y_pred = results[best_model]['predictions']
    
    axes[1, 0].scatter(y_test, y_pred, alpha=0.5)
    axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
    axes[1, 0].set_title(f'Actual vs Predicted - {best_model}', fontweight='bold')
    axes[1, 0].set_xlabel('Actual Burnout Score')
    axes[1, 0].set_ylabel('Predicted Burnout Score')
    axes[1, 0].legend()
    
    # Residuals plot
    residuals = y_test - y_pred
    axes[1, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_title(f'Residuals Plot - {best_model}', fontweight='bold')
    axes[1, 1].set_xlabel('Predicted Burnout Score')
    axes[1, 1].set_ylabel('Residuals')
    
    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: regression_results.png")


def main():
    """Main function for supervised learning."""
    print("Starting Supervised Learning Analysis...")
    
    # Load and preprocess data
    df, le_day, le_risk = load_and_preprocess_data()
    
    # Prepare features
    X, y_classification, y_regression, feature_names = prepare_features(df)
    
    # Classification models
    clf_results, X_test_clf, y_test_clf = classification_models(X, y_classification, feature_names)
    
    # Regression models
    reg_results, X_test_reg, y_test_reg = regression_models(X, y_regression, feature_names)
    
    print("\n" + "=" * 70)
    print("SUPERVISED LEARNING COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
