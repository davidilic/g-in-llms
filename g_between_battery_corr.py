import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
import pandas as pd
from factor_analyzer import calculate_kmo
import sys

def compute_g_factor(data, n_subtests=10, exclude_subtests=None):
    """
    Randomly selects n_subtests from the data, checks KMO, and performs principal axis factoring.
    Returns the g-factor (first principal axis) or None if KMO is below threshold.
    """
    available_subtests = data.columns if exclude_subtests is None else [col for col in data.columns if col not in exclude_subtests]
    selected_subtests = np.random.choice(available_subtests, size=n_subtests, replace=False)
    
    fa = FactorAnalysis(n_components=1, rotation=None)
    g_factor = fa.fit_transform(data[selected_subtests])
    
    return g_factor, selected_subtests

def correlate_g_factors(data, n_subtests=10):
    """
    Computes g-factors for two randomly selected sets of subtests and returns
    their correlation. If KMO is less than 0.9 for any set, return None.
    """
    g_factor_1, selected_subtests_1 = compute_g_factor(data, n_subtests)
    g_factor_2, selected_subtests_2 = compute_g_factor(data, n_subtests, exclude_subtests=selected_subtests_1)

    if g_factor_1 is None or g_factor_2 is None:
        return None
    
    correlation = np.corrcoef(g_factor_1.squeeze(), g_factor_2.squeeze())[0, 1]
    
    return correlation

def analyze_g_factor_correlations(data, n_successes=100):
    
    correlation_distribution = []
    while len(correlation_distribution) < n_successes:
        print(f"Number of successes: {len(correlation_distribution)}")
        correlation = correlate_g_factors(data)
        if correlation is not None: correlation_distribution.append(correlation)

    correlation_distribution = [corr for corr in correlation_distribution if corr is not None]

    # Summary statistics
    summary_stats = {
        'Mean': np.mean(correlation_distribution),
        'Median': np.median(correlation_distribution),
        'Standard Deviation': np.std(correlation_distribution),
    }

    print(correlation_distribution)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(correlation_distribution, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of correlation coefficients between scores on two g-factors derived from disjoint sets of subtests')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axvline(summary_stats['Mean'], color='red', linestyle='dashed', linewidth=1, label=f"Mean = {summary_stats['Mean']:.2f}")
    plt.axvline(summary_stats['Median'], color='green', linestyle='dashed', linewidth=1, label=f"Median = {summary_stats['Median']:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return summary_stats

data = pd.read_csv('./data/hf_leaderboard.csv', index_col=0)
data = data.drop(columns=['param_count'])
data = data.fillna(data.mean())

print(analyze_g_factor_correlations(data, 100))

