"""
Work From Home Burnout Analysis - Exploratory Data Analysis
This module performs comprehensive EDA on the WFH burnout dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_data(filepath='work_from_home_burnout_dataset.csv'):
    """Load the dataset."""
    df = pd.read_csv(filepath)
    return df


def data_quality_check(df):
    """Perform data quality checks."""
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumn Data Types:")
    print(df.dtypes)
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found!")
    else:
        print(missing[missing > 0])
    
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    return df


def visualize_distributions(df):
    """Visualize distributions of numerical features."""
    numerical_cols = ['work_hours', 'screen_time_hours', 'meetings_count', 
                      'breaks_taken', 'sleep_hours', 'task_completion_rate', 
                      'burnout_score']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        axes[idx].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    # Remove extra subplots
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('eda_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: eda_distributions.png")


def correlation_analysis(df):
    """Analyze correlations between features."""
    numerical_cols = ['work_hours', 'screen_time_hours', 'meetings_count', 
                      'breaks_taken', 'after_hours_work', 'sleep_hours', 
                      'task_completion_rate', 'burnout_score']
    
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: eda_correlation_matrix.png")
    
    print("\nTop Correlations with Burnout Score:")
    burnout_corr = correlation_matrix['burnout_score'].sort_values(ascending=False)
    print(burnout_corr)


def categorical_analysis(df):
    """Analyze categorical features."""
    print("\n" + "=" * 60)
    print("CATEGORICAL ANALYSIS")
    print("=" * 60)
    
    # Day type distribution
    print("\nDay Type Distribution:")
    print(df['day_type'].value_counts())
    
    # Burnout risk distribution
    print("\nBurnout Risk Distribution:")
    print(df['burnout_risk'].value_counts())
    
    # After hours work
    print("\nAfter Hours Work Distribution:")
    print(df['after_hours_work'].value_counts())
    
    # Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    df['day_type'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_title('Day Type Distribution')
    axes[0].set_ylabel('Count')
    
    df['burnout_risk'].value_counts().plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
    axes[1].set_title('Burnout Risk Distribution')
    axes[1].set_ylabel('Count')
    
    df['after_hours_work'].value_counts().plot(kind='bar', ax=axes[2], color='lightgreen', edgecolor='black')
    axes[2].set_title('After Hours Work Distribution')
    axes[2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('eda_categorical.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: eda_categorical.png")


def burnout_analysis(df):
    """Analyze burnout patterns."""
    print("\n" + "=" * 60)
    print("BURNOUT ANALYSIS")
    print("=" * 60)
    
    # Burnout by day type
    print("\nBurnout Score by Day Type:")
    print(df.groupby('day_type')['burnout_score'].describe())
    
    # Burnout by after hours work
    print("\nBurnout Score by After Hours Work:")
    print(df.groupby('after_hours_work')['burnout_score'].describe())
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Box plot: Burnout by day type
    df.boxplot(column='burnout_score', by='day_type', ax=axes[0, 0])
    axes[0, 0].set_title('Burnout Score by Day Type')
    axes[0, 0].set_xlabel('Day Type')
    axes[0, 0].set_ylabel('Burnout Score')
    
    # Box plot: Burnout by after hours work
    df.boxplot(column='burnout_score', by='after_hours_work', ax=axes[0, 1])
    axes[0, 1].set_title('Burnout Score by After Hours Work')
    axes[0, 1].set_xlabel('After Hours Work')
    axes[0, 1].set_ylabel('Burnout Score')
    
    # Scatter: Work hours vs Burnout
    axes[1, 0].scatter(df['work_hours'], df['burnout_score'], alpha=0.5)
    axes[1, 0].set_title('Work Hours vs Burnout Score')
    axes[1, 0].set_xlabel('Work Hours')
    axes[1, 0].set_ylabel('Burnout Score')
    
    # Scatter: Sleep hours vs Burnout
    axes[1, 1].scatter(df['sleep_hours'], df['burnout_score'], alpha=0.5, color='red')
    axes[1, 1].set_title('Sleep Hours vs Burnout Score')
    axes[1, 1].set_xlabel('Sleep Hours')
    axes[1, 1].set_ylabel('Burnout Score')
    
    plt.tight_layout()
    plt.savefig('eda_burnout_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: eda_burnout_analysis.png")


def main():
    """Main EDA function."""
    print("Starting Exploratory Data Analysis...")
    
    # Load data
    df = load_data()
    
    # Data quality check
    df = data_quality_check(df)
    
    # Visualize distributions
    visualize_distributions(df)
    
    # Correlation analysis
    correlation_analysis(df)
    
    # Categorical analysis
    categorical_analysis(df)
    
    # Burnout analysis
    burnout_analysis(df)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
