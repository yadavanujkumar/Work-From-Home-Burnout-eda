"""
Work From Home Burnout Analysis - Master Script
This script runs all analysis modules sequentially.
"""

import sys
import os

# Import all analysis modules
import eda_analysis
import supervised_learning
import timeseries_analysis
import workforce_hr_analytics


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    """Run all analysis modules."""
    print_header("WORK FROM HOME BURNOUT ANALYSIS - COMPREHENSIVE REPORT")
    
    print("This analysis covers:")
    print("  1. Exploratory Data Analysis (EDA)")
    print("  2. Supervised Learning (Classification & Regression)")
    print("  3. Time Series Trend Analysis")
    print("  4. Workforce & HR Analytics\n")
    
    try:
        # 1. Run EDA
        print_header("MODULE 1: EXPLORATORY DATA ANALYSIS")
        eda_analysis.main()
        
        # 2. Run Supervised Learning
        print_header("MODULE 2: SUPERVISED LEARNING")
        supervised_learning.main()
        
        # 3. Run Time Series Analysis
        print_header("MODULE 3: TIME SERIES ANALYSIS")
        timeseries_analysis.main()
        
        # 4. Run Workforce & HR Analytics
        print_header("MODULE 4: WORKFORCE & HR ANALYTICS")
        workforce_hr_analytics.main()
        
        # Final summary
        print_header("ANALYSIS COMPLETE")
        print("All analyses completed successfully!")
        print("\nGenerated Files:")
        print("  - eda_distributions.png")
        print("  - eda_correlation_matrix.png")
        print("  - eda_categorical.png")
        print("  - eda_burnout_analysis.png")
        print("  - classification_results.png")
        print("  - regression_results.png")
        print("  - timeseries_user_trends.png")
        print("  - timeseries_temporal_patterns.png")
        print("  - timeseries_burnout_progression.png")
        print("  - workforce_productivity_analysis.png")
        print("  - occupational_health_analysis.png")
        
        print("\nAll visualizations have been saved to the current directory.")
        print("Review the console output and generated images for insights.")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
