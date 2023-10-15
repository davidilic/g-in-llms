import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
#import statistics for correlation
from scipy.stats import pearsonr

def compute_correlated_g_loadings_efa(data, n_iterations=100, n_random_tasks=19):
    all_tasks = data.columns 
    correlated_g_loadings = []

    while len(correlated_g_loadings) < n_iterations:
        print(f"Iteration {len(correlated_g_loadings) + 1}/{n_iterations}")

        sampled_tasks = np.random.choice(all_tasks, n_random_tasks, replace=False).tolist()
        target_task = np.random.choice(sampled_tasks)

        other_tasks = [task for task in sampled_tasks if task != target_task]
        np.random.shuffle(other_tasks)
        battery1 = other_tasks[:9]
        battery2 = other_tasks[9:]
        
        battery1.append(target_task)
        battery2.append(target_task)

        def calc_g_loading(data, tasks, target_task):
            task_data = data[tasks]

            try:
                kmo_all, kmo_model = calculate_kmo(task_data)
            except:
                return None
            
            if kmo_model < 0.6:
                return None
            
            fa = FactorAnalyzer(rotation=None, method="principal")
            fa.fit(task_data)
            loadings = fa.loadings_[:, 0]
            target_task_index = tasks.index(target_task)
            return loadings[target_task_index]

        g_loading1 = calc_g_loading(data, battery1, target_task)
        g_loading2 = calc_g_loading(data, battery2, target_task)

        if g_loading1 is None or g_loading2 is None:
            continue
        correlated_g_loadings.append((g_loading1, g_loading2))

    return np.array(correlated_g_loadings)

data = pd.read_csv("./data/hf_leaderboard.csv")
data = data.drop(['model', 'param_count'], axis=1)
correlated_g_loadings = compute_correlated_g_loadings_efa(data)

correlation = pearsonr(correlated_g_loadings[:, 0], correlated_g_loadings[:, 1])
print(f"Correlation: {correlation[0]}, p-value: {correlation[1]}")

