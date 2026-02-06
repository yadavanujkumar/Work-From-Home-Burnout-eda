"""
Work From Home Burnout Analysis - Workforce & HR Analytics
This module provides workforce analytics and HR insights for occupational health.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath='work_from_home_burnout_dataset.csv'):
    """Load the dataset."""
    df = pd.read_csv(filepath)
    return df


def workforce_productivity_analysis(df):
    """Analyze workforce productivity metrics."""
    print("\n" + "=" * 70)
    print("WORKFORCE PRODUCTIVITY ANALYSIS")
    print("=" * 70)
    
    # Overall productivity metrics
    print("\nOverall Workforce Metrics:")
    print(f"Average Work Hours: {df['work_hours'].mean():.2f} hours")
    print(f"Average Screen Time: {df['screen_time_hours'].mean():.2f} hours")
    print(f"Average Meetings: {df['meetings_count'].mean():.2f} per day")
    print(f"Average Breaks: {df['breaks_taken'].mean():.2f} per day")
    print(f"Average Task Completion: {df['task_completion_rate'].mean():.2f}%")
    print(f"After Hours Work Rate: {(df['after_hours_work'].sum() / len(df) * 100):.2f}%")
    
    # Productivity by burnout risk
    print("\n" + "-" * 70)
    print("PRODUCTIVITY BY BURNOUT RISK LEVEL")
    print("-" * 70)
    
    productivity_by_risk = df.groupby('burnout_risk').agg({
        'work_hours': 'mean',
        'task_completion_rate': 'mean',
        'breaks_taken': 'mean',
        'meetings_count': 'mean',
        'after_hours_work': 'mean'
    }).round(2)
    
    print(productivity_by_risk)
    
    # Identify high-risk patterns
    high_risk = df[df['burnout_risk'] == 'High']
    print(f"\n\nHigh Risk Group Characteristics:")
    print(f"  - Size: {len(high_risk)} observations ({len(high_risk)/len(df)*100:.1f}%)")
    print(f"  - Avg Work Hours: {high_risk['work_hours'].mean():.2f}")
    print(f"  - Avg Sleep: {high_risk['sleep_hours'].mean():.2f} hours")
    print(f"  - Avg Task Completion: {high_risk['task_completion_rate'].mean():.2f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Work hours by risk
    risk_order = ['Low', 'Medium', 'High']
    sns.boxplot(data=df, x='burnout_risk', y='work_hours', order=risk_order, 
                ax=axes[0, 0], palette='RdYlGn_r')
    axes[0, 0].set_title('Work Hours by Burnout Risk', fontweight='bold')
    
    # Task completion by risk
    sns.boxplot(data=df, x='burnout_risk', y='task_completion_rate', order=risk_order,
                ax=axes[0, 1], palette='RdYlGn_r')
    axes[0, 1].set_title('Task Completion by Burnout Risk', fontweight='bold')
    
    # Breaks by risk
    sns.boxplot(data=df, x='burnout_risk', y='breaks_taken', order=risk_order,
                ax=axes[0, 2], palette='RdYlGn_r')
    axes[0, 2].set_title('Breaks Taken by Burnout Risk', fontweight='bold')
    
    # Meetings by risk
    sns.boxplot(data=df, x='burnout_risk', y='meetings_count', order=risk_order,
                ax=axes[1, 0], palette='RdYlGn_r')
    axes[1, 0].set_title('Meetings by Burnout Risk', fontweight='bold')
    
    # After hours work by risk
    after_hours_pct = df.groupby('burnout_risk')['after_hours_work'].mean() * 100
    after_hours_pct.reindex(risk_order).plot(kind='bar', ax=axes[1, 1], 
                                             color=['green', 'orange', 'red'],
                                             edgecolor='black')
    axes[1, 1].set_title('After Hours Work % by Risk', fontweight='bold')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].set_xticklabels(risk_order, rotation=0)
    
    # Screen time by risk
    sns.boxplot(data=df, x='burnout_risk', y='screen_time_hours', order=risk_order,
                ax=axes[1, 2], palette='RdYlGn_r')
    axes[1, 2].set_title('Screen Time by Burnout Risk', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('workforce_productivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: workforce_productivity_analysis.png")


def occupational_health_analysis(df):
    """Analyze occupational health metrics."""
    print("\n" + "=" * 70)
    print("OCCUPATIONAL HEALTH ANALYSIS")
    print("=" * 70)
    
    # Health indicators
    print("\nKey Health Indicators:")
    print(f"Average Sleep Hours: {df['sleep_hours'].mean():.2f}")
    print(f"% with Adequate Sleep (>7hrs): {(df['sleep_hours'] >= 7).sum() / len(df) * 100:.2f}%")
    print(f"% with Insufficient Sleep (<6hrs): {(df['sleep_hours'] < 6).sum() / len(df) * 100:.2f}%")
    
    # Work-life balance indicators
    print("\nWork-Life Balance Indicators:")
    print(f"% Working Overtime (>8hrs): {(df['work_hours'] > 8).sum() / len(df) * 100:.2f}%")
    print(f"% Working After Hours: {df['after_hours_work'].sum() / len(df) * 100:.2f}%")
    print(f"Average Breaks per Day: {df['breaks_taken'].mean():.2f}")
    
    # Risk factors
    df['sleep_deprived'] = df['sleep_hours'] < 6
    df['overworked'] = df['work_hours'] > 9
    df['insufficient_breaks'] = df['breaks_taken'] < 2
    df['excessive_screen_time'] = df['screen_time_hours'] > 10
    
    risk_factors = pd.DataFrame({
        'Risk Factor': ['Sleep Deprived (<6hrs)', 'Overworked (>9hrs)', 
                       'Insufficient Breaks (<2)', 'Excessive Screen Time (>10hrs)'],
        'Prevalence (%)': [
            df['sleep_deprived'].sum() / len(df) * 100,
            df['overworked'].sum() / len(df) * 100,
            df['insufficient_breaks'].sum() / len(df) * 100,
            df['excessive_screen_time'].sum() / len(df) * 100
        ]
    })
    
    print("\n" + "-" * 70)
    print("OCCUPATIONAL HEALTH RISK FACTORS")
    print("-" * 70)
    print(risk_factors)
    
    # Health by burnout risk
    health_by_risk = df.groupby('burnout_risk').agg({
        'sleep_hours': 'mean',
        'sleep_deprived': lambda x: x.sum() / len(x) * 100,
        'overworked': lambda x: x.sum() / len(x) * 100,
        'insufficient_breaks': lambda x: x.sum() / len(x) * 100
    }).round(2)
    health_by_risk.columns = ['Avg Sleep (hrs)', 'Sleep Deprived (%)', 
                              'Overworked (%)', 'Insufficient Breaks (%)']
    
    print("\n" + "-" * 70)
    print("HEALTH METRICS BY BURNOUT RISK")
    print("-" * 70)
    print(health_by_risk)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Sleep distribution
    axes[0, 0].hist(df['sleep_hours'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(7, color='green', linestyle='--', linewidth=2, label='Recommended (7hrs)')
    axes[0, 0].axvline(df['sleep_hours'].mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Average ({df["sleep_hours"].mean():.1f}hrs)')
    axes[0, 0].set_title('Sleep Hours Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Sleep Hours')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Risk factors prevalence
    risk_factors.plot(x='Risk Factor', y='Prevalence (%)', kind='barh', 
                     ax=axes[0, 1], legend=False, color='coral', edgecolor='black')
    axes[0, 1].set_title('Health Risk Factors Prevalence', fontweight='bold')
    axes[0, 1].set_xlabel('Prevalence (%)')
    
    # Sleep vs Burnout
    sns.scatterplot(data=df, x='sleep_hours', y='burnout_score', 
                   hue='burnout_risk', palette='RdYlGn_r', 
                   alpha=0.6, ax=axes[1, 0])
    axes[1, 0].set_title('Sleep vs Burnout Score', fontweight='bold')
    axes[1, 0].set_xlabel('Sleep Hours')
    axes[1, 0].set_ylabel('Burnout Score')
    
    # Work hours vs Sleep hours
    sns.scatterplot(data=df, x='work_hours', y='sleep_hours', 
                   hue='burnout_risk', palette='RdYlGn_r',
                   alpha=0.6, ax=axes[1, 1])
    axes[1, 1].set_title('Work Hours vs Sleep Hours', fontweight='bold')
    axes[1, 1].set_xlabel('Work Hours')
    axes[1, 1].set_ylabel('Sleep Hours')
    
    plt.tight_layout()
    plt.savefig('occupational_health_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: occupational_health_analysis.png")


def hr_recommendations(df):
    """Generate HR intervention recommendations."""
    print("\n" + "=" * 70)
    print("HR INTERVENTION RECOMMENDATIONS")
    print("=" * 70)
    
    # Identify at-risk employees
    high_risk_count = len(df[df['burnout_risk'] == 'High'])
    medium_risk_count = len(df[df['burnout_risk'] == 'Medium'])
    
    print(f"\nAt-Risk Population:")
    print(f"  High Risk: {high_risk_count} observations ({high_risk_count/len(df)*100:.1f}%)")
    print(f"  Medium Risk: {medium_risk_count} observations ({medium_risk_count/len(df)*100:.1f}%)")
    
    # Key intervention areas
    print("\n" + "-" * 70)
    print("KEY INTERVENTION AREAS")
    print("-" * 70)
    
    interventions = []
    
    # 1. Sleep improvement
    sleep_issues = df[df['sleep_hours'] < 6].shape[0]
    if sleep_issues > 0:
        interventions.append({
            'Area': 'Sleep Health',
            'Priority': 'High',
            'Affected': f"{sleep_issues} ({sleep_issues/len(df)*100:.1f}%)",
            'Recommendation': 'Implement sleep hygiene programs, flexible work hours'
        })
    
    # 2. Workload management
    overwork_issues = df[df['work_hours'] > 9].shape[0]
    if overwork_issues > 0:
        interventions.append({
            'Area': 'Workload Management',
            'Priority': 'High',
            'Affected': f"{overwork_issues} ({overwork_issues/len(df)*100:.1f}%)",
            'Recommendation': 'Review workload distribution, enforce work hour limits'
        })
    
    # 3. Break policies
    break_issues = df[df['breaks_taken'] < 2].shape[0]
    if break_issues > 0:
        interventions.append({
            'Area': 'Break Policies',
            'Priority': 'Medium',
            'Affected': f"{break_issues} ({break_issues/len(df)*100:.1f}%)",
            'Recommendation': 'Mandate regular breaks, create break-friendly culture'
        })
    
    # 4. Screen time
    screen_issues = df[df['screen_time_hours'] > 10].shape[0]
    if screen_issues > 0:
        interventions.append({
            'Area': 'Screen Time',
            'Priority': 'Medium',
            'Affected': f"{screen_issues} ({screen_issues/len(df)*100:.1f}%)",
            'Recommendation': 'Promote 20-20-20 rule, encourage screen breaks'
        })
    
    # 5. After hours work
    after_hours_issues = df[df['after_hours_work'] == 1].shape[0]
    if after_hours_issues > 0:
        interventions.append({
            'Area': 'Work-Life Balance',
            'Priority': 'High',
            'Affected': f"{after_hours_issues} ({after_hours_issues/len(df)*100:.1f}%)",
            'Recommendation': 'Establish clear boundaries, discourage after-hours communication'
        })
    
    intervention_df = pd.DataFrame(interventions)
    print(intervention_df.to_string(index=False))
    
    # Additional insights
    print("\n" + "-" * 70)
    print("ADDITIONAL INSIGHTS")
    print("-" * 70)
    
    # Correlation insights
    corr_with_burnout = df[[
        'work_hours', 'screen_time_hours', 'meetings_count', 
        'breaks_taken', 'after_hours_work', 'sleep_hours', 
        'task_completion_rate', 'burnout_score'
    ]].corr()['burnout_score'].sort_values(ascending=False)
    
    print("\nTop Factors Contributing to Burnout:")
    print(corr_with_burnout.head(5))
    
    print("\nTop Protective Factors Against Burnout:")
    print(corr_with_burnout.tail(5))


def generate_summary_report(df):
    """Generate executive summary report."""
    print("\n" + "=" * 70)
    print("EXECUTIVE SUMMARY REPORT")
    print("=" * 70)
    
    total_observations = len(df)
    unique_users = df['user_id'].nunique()
    
    print(f"\nDataset Overview:")
    print(f"  Total Observations: {total_observations}")
    print(f"  Unique Users: {unique_users}")
    print(f"  Average Observations per User: {total_observations/unique_users:.1f}")
    
    print(f"\nBurnout Distribution:")
    risk_dist = df['burnout_risk'].value_counts()
    for risk, count in risk_dist.items():
        print(f"  {risk} Risk: {count} ({count/total_observations*100:.1f}%)")
    
    print(f"\nKey Workforce Metrics:")
    print(f"  Average Burnout Score: {df['burnout_score'].mean():.2f}")
    print(f"  Average Work Hours: {df['work_hours'].mean():.2f}")
    print(f"  Average Sleep Hours: {df['sleep_hours'].mean():.2f}")
    print(f"  Average Task Completion: {df['task_completion_rate'].mean():.2f}%")
    print(f"  After Hours Work Rate: {df['after_hours_work'].mean()*100:.2f}%")
    
    print(f"\nCritical Findings:")
    print(f"  ⚠ {(df['sleep_hours'] < 6).sum()} observations with severe sleep deprivation")
    print(f"  ⚠ {(df['work_hours'] > 9).sum()} observations with excessive work hours")
    print(f"  ⚠ {(df['breaks_taken'] < 2).sum()} observations with insufficient breaks")
    print(f"  ⚠ {(df['burnout_score'] > 70).sum()} observations with high burnout scores")


def main():
    """Main function for workforce and HR analytics."""
    print("Starting Workforce & HR Analytics...")
    
    # Load data
    df = load_data()
    
    # Workforce productivity analysis
    workforce_productivity_analysis(df)
    
    # Occupational health analysis
    occupational_health_analysis(df)
    
    # HR recommendations
    hr_recommendations(df)
    
    # Summary report
    generate_summary_report(df)
    
    print("\n" + "=" * 70)
    print("WORKFORCE & HR ANALYTICS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
