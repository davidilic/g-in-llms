import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.stats import pearsonr, linregress
import matplotlib.pyplot as plt

class GeneralIntelligenceAnalysis:
    def __init__(self, dataset_path, subtests):
        self.dataset_path = dataset_path
        self.subtests = subtests
        self.data = pd.read_csv(dataset_path).set_index('model')

    def preprocess(self):
        relevant_cols = self.subtests + ['param_count']
        self.data = self.data[relevant_cols]
        self.data.fillna(self.data.median(), inplace=True)

    def derive_general_intelligence(self):
        factor_analysis = FactorAnalysis(n_components=1)
        self.data['general_intelligence'] = factor_analysis.fit_transform(self.data[self.subtests])

    def correlate_with_param_count(self):
        correlation, p_value = pearsonr(self.data['param_count'], self.data['general_intelligence'])
        return correlation, p_value

    def plot_correlation(self):
        plt.scatter(self.data['param_count'], self.data['general_intelligence'])
        
        # Add the regression line
        slope, intercept, _, _, _ = linregress(self.data['param_count'], self.data['general_intelligence'])
        plt.plot(self.data['param_count'], intercept + slope * self.data['param_count'], 'r')
        
        plt.xlabel('Parameter Count')
        plt.ylabel('General Intelligence')
        plt.title('Correlation between Parameter Count and General Intelligence')
        plt.show()

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

analysis = GeneralIntelligenceAnalysis("./data/hf_leaderboard.csv", subtests)
analysis.preprocess()
analysis.derive_general_intelligence()
correlation, p_value = analysis.correlate_with_param_count()
print(f"Correlation: {correlation}, P-value: {p_value}")
analysis.plot_correlation()
