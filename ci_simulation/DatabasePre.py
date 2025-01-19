from ci_simulation.DatabaseBaseMock import DatabaseBaseMock


class DatabasePre(DatabaseBaseMock):

    def __init__(self, filename, evaluator, group_key='build', history=5, timestamp=0):
        """
        Database mock for check processing in CI simulation.

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
            'last_failure': -1,
            'historical_executions': 0
        }
        super().__init__(filename, evaluator, default_test_history, group_key=group_key, history=history, timestamp=timestamp)

    def init_job_metrics(self, job_data):
        """
        Add metadata for evaluation such as hypothetical reward (for ideal execution).

        Parameters:
        job_data (pd.DataFrame): DataFrame of CI job runs.

        Returns:
        pd.DataFrame: DataFrame with added hypothetical_reward column.
        """
        job_data['hypothetical_reward'] = job_data.apply(lambda row: self.reward_function(row), axis=1)
        return job_data

    @staticmethod
    def reward_function(test_item):
        """
        Reward function. Reward is one if test execution failed, otherwise zero.

        Parameters:
        test_item (pd.Series): Series containing the details of the current test execution.

        Returns:
        int: Reward value.
        """
        return 1 if test_item.test_result == "FAILED" else 0

    def execute_tests(self, scheduled_tests):
        """
        Simulate the execution of the selected test cases and update the historical execution database.

        Parameters:
        scheduled_tests (list): List of test names that are scheduled for execution.

        Returns:
        list: Rewards corresponding to the information gain for each executed test.
        """
        rewards = [-1 for _ in range(len(scheduled_tests))]
        job_data = self.current_env

        cycle_duration = 0.0
        full_cycle_duration = job_data.test_duration.sum()
        for _, test_item in job_data.iterrows():
            test_name = test_item.test_name

            # Test was not scheduled
            if test_name not in scheduled_tests:
                self.test_history[test_name]['last_execution'] += 1
                continue

            reward = float(1 - 0.9 * cycle_duration / full_cycle_duration)
            if self.reward_function(test_item) == 0:
                reward = -float(1 - 0.9 * cycle_duration / full_cycle_duration)
            cycle_duration += test_item.test_duration
            rewards[scheduled_tests.index(test_name)] = reward

            if reward > 0:
                self.evaluator.log_execution_results({'test_name': test_name, 'reward': reward, 'time': self.time})

            # Update test history
            for i in range(0, self.history - 1):
                self.test_history[test_name][f'result_t-{self.history - i}'] = self.test_history[test_name][
                    f'result_t-{self.history - (i + 1)}']
                self.test_history[test_name][f'exec_t-{self.history - i}'] = self.test_history[test_name][
                    f'exec_t-{self.history - (i + 1)}']

            self.test_history[test_name]['result_t-1'] = test_item['test_result']
            self.test_history[test_name]['exec_t-1'] = self.time
            self.test_history[test_name]['hypothetical_reward'] = 0

            # Update last failure
            if test_item['test_result'] == 'FAILED':
                self.test_history[test_name]['last_failure'] = 0
            else:
                self.test_history[test_name]['last_failure'] += 1

            # Update last execution
            self.test_history[test_name]['last_execution'] = 0
            self.test_history[test_name]['historical_executions'] += 0.5

            # Update test duration
            self.test_history[test_name]['avg_test_duration'] = test_item['test_duration']

        self.time += 1
        return rewards
