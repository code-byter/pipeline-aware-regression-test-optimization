import pandas as pd

class DatabaseBaseMock:
    job_data = {}
    test_history = {}
    time = 0
    current_env = pd.DataFrame()

    def __init__(self, filename, evaluator, default_test_history, group_key='start_time', history=5, timestamp=0):
        """
        Base class for database mocks used in CI simulations.

        Parameters:
        filename (str): Path to the dataset file.
        evaluator (Evaluator): Evaluator object for logging and evaluation purposes.
        default_test_history (dict): Default test history values.
        group_key (str): Key for grouping the data. Defaults to 'start_time'.
        history (int): Number of historical test results to maintain. Defaults to 5.
        timestamp (int): Initial timestamp. Defaults to 0.
        """
        # Load historical data from logs
        df = pd.read_pickle(filename)

        self.evaluator = evaluator
        self.history = history
        self.time = timestamp

        # Group dataset by job executions
        df = df.sort_values(by=[group_key, 'test_duration'])
        self.job_data = dict(tuple(df.groupby(group_key)))

        # Initialize history by value NO_STATUS
        for i in range(self.history):
            default_test_history[f'result_t-{i + 1}'] = 'NO_STATUS'
            default_test_history[f'exec_t-{i + 1}'] = 0

        # Create table for historical data
        for test_name in df.test_name.unique():
            self.test_history[test_name] = default_test_history.copy()
            self.test_history[test_name]['test_name'] = test_name

        print(f"Loaded dataframe of length {len(df)} with {len(self.job_data)} builds")

    def get_job(self, time):
        """
        Loads test data from CI logs and historical execution data and merges them.
        The final dataframe is stored in self.current_env.

        Parameters:
        time (int): Timestamp for observation.

        Returns:
        pd.DataFrame: DataFrame with all information on the test cases of the current job.
        """
        if not self.set_timestamp(time):
            return pd.DataFrame()

        self.current_env = pd.DataFrame()

        # Load CI logs and historical data for job run at the specified time
        job_data = self.job_data[list(self.job_data.keys())[self.time]].copy()
        historical_data = pd.DataFrame([self.test_history[test_name] for test_name in job_data.test_name.unique()])

        # Encode strings
        job_data.test_name = job_data.test_name.str.encode('utf-8')
        historical_data.test_name = historical_data.test_name.str.encode('utf-8')

        # Merge historical data and CI logs
        job_data = pd.merge(job_data, historical_data, on='test_name')
        job_data.test_name = job_data.test_name.str.decode('utf-8')

        # Add custom metrics such as information gain
        job_data = self.init_job_metrics(job_data)
        job_data = job_data.sort_values(by=['test_duration'], ascending=False)

        # Check for duplicate data in builds
        if len(job_data) != len(historical_data):
            print("Error: Build contains duplicate data")
            job_data = pd.DataFrame()
        self.current_env = job_data

        return self.current_env

    def execute_tests(self, scheduled_tests):
        """
        Simulates the execution of the selected test cases and updates the historical execution database.

        Parameters:
        scheduled_tests (list): List of test names that are scheduled for execution.

        Returns:
        tuple: Rewards and additional information (dummy implementation).
        """
        rewards = [0 for _ in range(len(scheduled_tests))]
        info = []
        job_data = self.current_env

        for test_name in job_data.test_name.to_list():
            if test_name in scheduled_tests:
                continue

        self.time += 1
        return rewards, info

    def init_job_metrics(self, job_data):
        """
        Initialize job metrics. Dummy implementation.

        Parameters:
        job_data (pd.DataFrame): DataFrame of job data.

        Returns:
        pd.DataFrame: DataFrame with initialized metrics.
        """
        return job_data

    def set_timestamp(self, time):
        """
        Set the timestamp of the fetched job data.

        Parameters:
        time (int): Timestamp to set.

        Returns:
        bool: True if timestamp is valid, False otherwise.
        """
        if self.time >= len(self.job_data):
            print(f"Error: ES data for time {self.time} not available")
            print("Restarting from t=0")
            return False
        self.time = time
        return True

    def get_feature_names(self):
        """
        Get the feature names from the job data.

        Returns:
        list: List of feature names.
        """
        return list(self.job_data[list(self.job_data.keys())[0]].columns)

    def log_current_execution_delays(self):
        """
        Log the current execution delays for all tests.
        """
        for test_name in list(self.test_history.keys()):
            if self.test_history[test_name]['execution_delay'] > 0:
                self.evaluator.log_execution_results({
                    'test_name': test_name,
                    'information_gain': -1,
                    'execution_delay': self.test_history[test_name]['execution_delay']
                })
