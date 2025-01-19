from ci_simulation.DatabaseBaseMock import DatabaseBaseMock
from helpers.information_gain import compute_information_gain_non_flaky

class DatabasePost(DatabaseBaseMock):
    def __init__(self, filename, evaluator, group_key='start_time', history=5, timestamp=0):
        """
        Database mock for post-processing CI simulation data.

        Parameters:
        filename (str): Path to the dataset file.
        evaluator (Evaluator): Evaluator object for logging and evaluation purposes.
        group_key (str): Key for grouping the data. Defaults to 'start_time'.
        history (int): Number of historical test results to maintain. Defaults to 5.
        timestamp (int): Initial timestamp. Defaults to 0.
        """
        default_test_history = {
            'hypothetical_reward': 0,
            'avg_test_duration': 0.5,
            'last_execution': -1,
            'last_verdict': -1,
            'historical_executions': 0,
            'execution_delay': 0
        }
        super().__init__(filename, evaluator, default_test_history, history=history, timestamp=timestamp, group_key='build')

    def init_job_metrics(self, job_data):
        """
        Add metadata for evaluation such as hypothetical reward for ideal execution.

        Parameters:
        job_data (pd.DataFrame): DataFrame of CI job runs.

        Returns:
        pd.DataFrame: DataFrame with added hypothetical_reward column.
        """
        job_data['hypothetical_reward'] = job_data.apply(lambda row: compute_information_gain_non_flaky(row, row), axis=1)
        return job_data

    def execute_tests(self, scheduled_tests):
        """
        Simulate the execution of the selected test cases and updates the historical execution database.

        Parameters:
        scheduled_tests (list): List of test names that are scheduled for execution.

        Returns:
        list: Rewards corresponding to the information gain for each executed test.
        """
        rewards = [-1 for _ in range(len(scheduled_tests))]
        job_data = self.current_env

        for index, test_item in job_data.iterrows():
            test_name = test_item.test_name
            reward = compute_information_gain_non_flaky(test_item, self.test_history[test_name])

            # Test was not scheduled
            if test_name not in scheduled_tests:
                self.test_history[test_name]['last_execution'] += 1
                if reward > 0:
                    self.test_history[test_name]['execution_delay'] += 1
                continue

            rewards[scheduled_tests.index(test_name)] = reward

            # Log execution data if reward is positive
            if reward > 0:
                self.evaluator.log_execution_results({
                    'test_name': test_name,
                    'reward': reward,
                    'time': self.time,
                    'execution_delay': self.test_history[test_name]['execution_delay']
                })
                self.test_history[test_name]['last_verdict'] = 0
            else:
                self.test_history[test_name]['last_verdict'] = 0

            # Update test history
            for i in range(0, self.history - 1):
                self.test_history[test_name][f'result_t-{self.history - i}'] = self.test_history[test_name][f'result_t-{self.history - (i + 1)}']
                self.test_history[test_name][f'exec_t-{self.history - i}'] = self.test_history[test_name][f'exec_t-{self.history - (i + 1)}']

            self.test_history[test_name]['result_t-1'] = test_item['test_result']
            self.test_history[test_name]['exec_t-1'] = self.time
            self.test_history[test_name]['execution_delay'] = 0
            self.test_history[test_name]['last_execution'] = 0
            self.test_history[test_name]['hypothetical_reward'] = 0

            # Update last execution
            self.test_history[test_name]['historical_executions'] += 0.5

            # Update test duration
            self.test_history[test_name]['avg_test_duration'] = test_item['test_duration']

        self.time += 1
        return rewards
