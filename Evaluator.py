import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Evaluator:
    job_results = []
    execution_results = []
    training_history = []

    def __init__(self, name):
        self.name = name

    def log_job_results(self, time, metrics):
        metrics['name'] = self.name
        metrics['time'] = time
        self.job_results.append(metrics)

    def log_execution_results(self, execution_results):
        execution_results['name'] = self.name
        self.execution_results.append(execution_results)

    def save(self, config):
        job_results = pd.DataFrame(self.job_results)
        job_results.to_pickle(f"results/job_results_{config}.pkl")

