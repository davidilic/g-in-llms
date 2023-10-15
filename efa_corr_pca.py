import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis, PCA
from scipy.stats import pearsonr

subtests = [
    "arc_challenge",
    "hellaswag",
    "business_ethics",
    "clinical_knowledge",
    "college_computer_science",
    "college_mathematics",
    "computer_security",
    "conceptual_physics",
    "global_facts",
    "hs_computer_science",
    "hs_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "philosophy",
    "professional_medicine",
    "us_foreign_policy",
    "virology",
    "world_religions",
    "truthfulqa_mc"
]

def compute_principal_vectors(data):
    fa = FactorAnalysis(n_components=1)
    fa_scores = fa.fit_transform(data)
    
    pca = PCA(n_components=1)
    pca_scores = pca.fit_transform(data)
    
    return fa_scores, pca_scores

def compute_correlation(fa_scores, pca_scores):
    correlation_coefficient, p_value = pearsonr(fa_scores.squeeze(), pca_scores.squeeze())
    return correlation_coefficient, p_value

data = pd.read_csv('./data/hf_leaderboard.csv')

data = data[subtests]
data = data.fillna(data.mean())

fa_scores, pca_scores = compute_principal_vectors(data)

correlation_coefficient, p_value = compute_correlation(fa_scores, pca_scores)
print(f"Correlation Coefficient: {correlation_coefficient}, P-value: {p_value}")
