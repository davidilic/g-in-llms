import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from scipy.stats import kendalltau
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
        correlation, p_value = kendalltau(self.data['param_count'], self.data['general_intelligence'])
        return correlation, p_value

    def bootstrap_kendall_tau_ci(self, num_bootstrap_samples=1000, confidence_level=0.95):
        bootstrap_samples = np.zeros(num_bootstrap_samples)
        for i in range(num_bootstrap_samples):
            resample = self.data.sample(n=len(self.data), replace=True)
            bootstrap_samples[i] = kendalltau(resample['param_count'], resample['general_intelligence'])[0]

        lower_percentile = ((1 - confidence_level) / 2) * 100
        upper_percentile = (1 - (1 - confidence_level) / 2) * 100
        ci_lower = np.percentile(bootstrap_samples, lower_percentile)
        ci_upper = np.percentile(bootstrap_samples, upper_percentile)

        return ci_lower, ci_upper

    def plot_correlation(self):
        plt.scatter(self.data['param_count'], self.data['general_intelligence'])
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

ci_lower, ci_upper = analysis.bootstrap_kendall_tau_ci()
print(f"95% CI for Kendall's tau: [{ci_lower}, {ci_upper}]")

analysis.plot_correlation()