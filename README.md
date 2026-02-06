# Work From Home Burnout Analysis

Comprehensive analysis of work-from-home burnout patterns using machine learning, time-series analysis, and workforce analytics.

## 📊 Project Overview

This project analyzes a dataset of 1,800+ observations tracking work-from-home patterns and burnout levels. The analysis includes:

- **Exploratory Data Analysis (EDA)**: Understanding data distributions, correlations, and patterns
- **Supervised Learning**: Classification and regression models for burnout prediction
- **Time-Series Analysis**: Temporal trends and burnout progression patterns
- **Workforce Analytics**: Productivity metrics and work patterns
- **HR Analytics**: Occupational health insights and intervention recommendations

## 📁 Dataset Description

The dataset (`work_from_home_burnout_dataset.csv`) contains the following features:

- `user_id`: Unique identifier for each employee
- `day_type`: Weekday or Weekend
- `work_hours`: Daily work hours
- `screen_time_hours`: Daily screen time
- `meetings_count`: Number of meetings per day
- `breaks_taken`: Number of breaks per day
- `after_hours_work`: Binary indicator (0/1)
- `sleep_hours`: Hours of sleep
- `task_completion_rate`: Percentage of tasks completed
- `burnout_score`: Continuous burnout score (0-100)
- `burnout_risk`: Categorical risk level (Low/Medium/High)

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yadavanujkumar/Work-From-Home-Burnout-eda.git
cd Work-From-Home-Burnout-eda
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Analysis

**Option 1: Run all analyses at once**
```bash
python run_all_analysis.py
```

**Option 2: Run individual modules**
```bash
# Exploratory Data Analysis
python eda_analysis.py

# Supervised Learning Models
python supervised_learning.py

# Time Series Analysis
python timeseries_analysis.py

# Workforce & HR Analytics
python workforce_hr_analytics.py
```

## 📈 Analysis Modules

### 1. Exploratory Data Analysis (`eda_analysis.py`)
- Data quality checks and validation
- Distribution analysis of all features
- Correlation matrix and feature relationships
- Categorical variable analysis
- Burnout pattern identification

**Outputs:**
- `eda_distributions.png`
- `eda_correlation_matrix.png`
- `eda_categorical.png`
- `eda_burnout_analysis.png`

### 2. Supervised Learning (`supervised_learning.py`)

#### Classification Models (Burnout Risk Prediction)
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

**Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix

#### Regression Models (Burnout Score Prediction)
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

**Metrics:** R² Score, RMSE, MAE

**Outputs:**
- `classification_results.png`
- `regression_results.png`

### 3. Time Series Analysis (`timeseries_analysis.py`)
- User-level burnout trend analysis
- Temporal pattern analysis (Weekday vs Weekend)
- Burnout progression over time
- Risk level distribution trends

**Outputs:**
- `timeseries_user_trends.png`
- `timeseries_temporal_patterns.png`
- `timeseries_burnout_progression.png`

### 4. Workforce & HR Analytics (`workforce_hr_analytics.py`)
- Workforce productivity metrics
- Occupational health risk factors
- Work-life balance indicators
- HR intervention recommendations
- Executive summary report

**Outputs:**
- `workforce_productivity_analysis.png`
- `occupational_health_analysis.png`

## 🔍 Key Findings

### Burnout Risk Factors
- **Work Hours**: Longer work hours strongly correlate with higher burnout
- **Sleep Deprivation**: Less than 6 hours of sleep significantly increases risk
- **Insufficient Breaks**: Fewer than 2 breaks per day correlates with burnout
- **After Hours Work**: Working after hours is a significant risk factor
- **Screen Time**: Excessive screen time (>10 hours) contributes to burnout

### Protective Factors
- **Adequate Sleep**: 7+ hours of sleep reduces burnout risk
- **Regular Breaks**: More frequent breaks correlate with lower burnout
- **Task Completion**: Higher task completion rates indicate better work management

## 🎯 HR Recommendations

1. **Sleep Health Programs**: Implement sleep hygiene education and flexible work hours
2. **Workload Management**: Review and redistribute excessive workloads
3. **Break Policies**: Mandate regular breaks and create a break-friendly culture
4. **Work-Life Boundaries**: Establish clear boundaries for after-hours work
5. **Screen Time Management**: Promote the 20-20-20 rule and screen breaks
6. **Meeting Optimization**: Reduce unnecessary meetings to free up productive time

## 📊 Model Performance

### Classification (Burnout Risk)
- Best model accuracy: ~85-90% (varies by model)
- Feature importance: Sleep hours, work hours, and task completion are top predictors

### Regression (Burnout Score)
- Best model R² Score: ~0.85-0.90
- RMSE: ~5-8 points on burnout score scale

## 🛠️ Technologies Used

- **Python 3.x**
- **Data Analysis**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn
- **Statistics**: scipy

## 📝 Use Cases

This analysis is valuable for:
- **HR Departments**: Identify at-risk employees and design interventions
- **Occupational Health**: Monitor workplace health indicators
- **Management**: Understand productivity patterns and optimize workload
- **Researchers**: Study work-from-home burnout patterns
- **Data Scientists**: Learn ML applications in workforce analytics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This analysis is for educational and research purposes. Always consult with HR professionals and occupational health experts when making workplace decisions.