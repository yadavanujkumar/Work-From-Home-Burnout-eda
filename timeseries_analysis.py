"""
Work From Home Burnout Analysis - Time Series Analysis
This module analyzes temporal trends and patterns in burnout data.
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


def user_level_analysis(df):
    """Analyze burnout trends at user level."""
    print("\n" + "=" * 70)
    print("USER-LEVEL TIME SERIES ANALYSIS")
    print("=" * 70)
    
    # Group by user and calculate progression metrics
    user_stats = df.groupby('user_id').agg({
        'burnout_score': ['mean', 'std', 'min', 'max', 'count'],
        'work_hours': 'mean',
        'sleep_hours': 'mean',
        'breaks_taken': 'mean',
        'task_completion_rate': 'mean'
    }).round(2)
    
    print("\nUser Statistics Summary:")
    print(user_stats.head(10))
    
    # Calculate burnout progression (trend)
    user_trends = []
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].reset_index(drop=True)
        if len(user_data) > 1:
            # Calculate linear trend
            x = np.arange(len(user_data))
            y = user_data['burnout_score'].values
            if len(x) > 1 and not np.all(y == y[0]):
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                user_trends.append({
                    'user_id': user_id,
                    'trend_slope': slope,
                    'r_squared': r_value**2,
                    'initial_burnout': y[0],
                    'final_burnout': y[-1],
                    'burnout_change': y[-1] - y[0],
                    'observations': len(user_data)
                })
    
    trends_df = pd.DataFrame(user_trends)
    
    print(f"\nTotal users analyzed: {len(trends_df)}")
    print(f"\nUsers with increasing burnout (positive slope): {(trends_df['trend_slope'] > 0).sum()}")
    print(f"Users with decreasing burnout (negative slope): {(trends_df['trend_slope'] < 0).sum()}")
    
    print("\nTop 5 Users with Highest Burnout Increase:")
    print(trends_df.nlargest(5, 'burnout_change')[['user_id', 'initial_burnout', 
                                                      'final_burnout', 'burnout_change']])
    
    print("\nTop 5 Users with Highest Burnout Decrease:")
    print(trends_df.nsmallest(5, 'burnout_change')[['user_id', 'initial_burnout', 
                                                       'final_burnout', 'burnout_change']])
    
    return trends_df


def plot_user_trends(df, trends_df, num_users=6):
    """Plot burnout trends for selected users."""
    # Select users with varying trends
    high_increase = trends_df.nlargest(2, 'burnout_change')['user_id'].values
    high_decrease = trends_df.nsmallest(2, 'burnout_change')['user_id'].values
    moderate = trends_df.iloc[len(trends_df)//2:len(trends_df)//2+2]['user_id'].values
    
    selected_users = list(high_increase) + list(high_decrease) + list(moderate)
    selected_users = selected_users[:num_users]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, user_id in enumerate(selected_users):
        user_data = df[df['user_id'] == user_id].reset_index(drop=True)
        
        axes[idx].plot(user_data.index, user_data['burnout_score'], 
                      marker='o', linewidth=2, markersize=6)
        
        # Add trend line
        x = np.arange(len(user_data))
        y = user_data['burnout_score'].values
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axes[idx].plot(x, p(x), "r--", alpha=0.7, linewidth=2, label='Trend')
        
        axes[idx].set_title(f'User {user_id} - Burnout Progression', fontweight='bold')
        axes[idx].set_xlabel('Time Period')
        axes[idx].set_ylabel('Burnout Score')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig('timeseries_user_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: timeseries_user_trends.png")


def temporal_patterns(df):
    """Analyze temporal patterns (weekday vs weekend)."""
    print("\n" + "=" * 70)
    print("TEMPORAL PATTERNS ANALYSIS")
    print("=" * 70)
    
    # Compare weekday vs weekend
    temporal_comparison = df.groupby('day_type').agg({
        'burnout_score': ['mean', 'std'],
        'work_hours': 'mean',
        'screen_time_hours': 'mean',
        'meetings_count': 'mean',
        'breaks_taken': 'mean',
        'after_hours_work': 'mean',
        'sleep_hours': 'mean',
        'task_completion_rate': 'mean'
    }).round(2)
    
    print("\nWeekday vs Weekend Comparison:")
    print(temporal_comparison)
    
    # Statistical test
    weekday_burnout = df[df['day_type'] == 'Weekday']['burnout_score']
    weekend_burnout = df[df['day_type'] == 'Weekend']['burnout_score']
    
    t_stat, p_value = stats.ttest_ind(weekday_burnout, weekend_burnout)
    print(f"\nT-test Results (Weekday vs Weekend Burnout):")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("Significant difference found between weekday and weekend burnout!")
    else:
        print("No significant difference found.")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Burnout by day type
    df.boxplot(column='burnout_score', by='day_type', ax=axes[0, 0])
    axes[0, 0].set_title('Burnout Score: Weekday vs Weekend')
    axes[0, 0].set_xlabel('Day Type')
    axes[0, 0].set_ylabel('Burnout Score')
    
    # Work hours by day type
    df.boxplot(column='work_hours', by='day_type', ax=axes[0, 1])
    axes[0, 1].set_title('Work Hours: Weekday vs Weekend')
    axes[0, 1].set_xlabel('Day Type')
    axes[0, 1].set_ylabel('Work Hours')
    
    # Sleep hours by day type
    df.boxplot(column='sleep_hours', by='day_type', ax=axes[1, 0])
    axes[1, 0].set_title('Sleep Hours: Weekday vs Weekend')
    axes[1, 0].set_xlabel('Day Type')
    axes[1, 0].set_ylabel('Sleep Hours')
    
    # Task completion by day type
    df.boxplot(column='task_completion_rate', by='day_type', ax=axes[1, 1])
    axes[1, 1].set_title('Task Completion Rate: Weekday vs Weekend')
    axes[1, 1].set_xlabel('Day Type')
    axes[1, 1].set_ylabel('Task Completion Rate (%)')
    
    plt.tight_layout()
    plt.savefig('timeseries_temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: timeseries_temporal_patterns.png")


def burnout_progression_analysis(df):
    """Analyze overall burnout progression patterns."""
    print("\n" + "=" * 70)
    print("BURNOUT PROGRESSION PATTERNS")
    print("=" * 70)
    
    # Aggregate data by observation sequence
    # For each user, track their burnout journey
    progression_data = []
    
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].reset_index(drop=True)
        for idx, row in user_data.iterrows():
            progression_data.append({
                'user_id': user_id,
                'sequence': idx,
                'burnout_score': row['burnout_score'],
                'burnout_risk': row['burnout_risk']
            })
    
    prog_df = pd.DataFrame(progression_data)
    
    # Average burnout by sequence position
    avg_by_sequence = prog_df.groupby('sequence')['burnout_score'].agg(['mean', 'std', 'count'])
    
    print("\nAverage Burnout Score by Observation Sequence:")
    print(avg_by_sequence.head(10))
    
    # Plot progression
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Average burnout progression
    axes[0].plot(avg_by_sequence.index, avg_by_sequence['mean'], 
                marker='o', linewidth=2, markersize=8, color='red')
    axes[0].fill_between(avg_by_sequence.index, 
                        avg_by_sequence['mean'] - avg_by_sequence['std'],
                        avg_by_sequence['mean'] + avg_by_sequence['std'],
                        alpha=0.3, color='red')
    axes[0].set_title('Average Burnout Progression Over Time', fontweight='bold')
    axes[0].set_xlabel('Observation Sequence')
    axes[0].set_ylabel('Burnout Score')
    axes[0].grid(True, alpha=0.3)
    
    # Risk level distribution over time
    risk_by_seq = prog_df.groupby(['sequence', 'burnout_risk']).size().unstack(fill_value=0)
    risk_by_seq_pct = risk_by_seq.div(risk_by_seq.sum(axis=1), axis=0) * 100
    
    risk_by_seq_pct.plot(kind='area', stacked=True, ax=axes[1], 
                         alpha=0.7, color=['green', 'orange', 'red'])
    axes[1].set_title('Burnout Risk Distribution Over Time', fontweight='bold')
    axes[1].set_xlabel('Observation Sequence')
    axes[1].set_ylabel('Percentage (%)')
    axes[1].legend(title='Risk Level')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timeseries_burnout_progression.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\nSaved: timeseries_burnout_progression.png")


def main():
    """Main function for time series analysis."""
    print("Starting Time Series Analysis...")
    
    # Load data
    df = load_data()
    
    # User-level analysis
    trends_df = user_level_analysis(df)
    
    # Plot user trends
    plot_user_trends(df, trends_df)
    
    # Temporal patterns
    temporal_patterns(df)
    
    # Burnout progression
    burnout_progression_analysis(df)
    
    print("\n" + "=" * 70)
    print("TIME SERIES ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
