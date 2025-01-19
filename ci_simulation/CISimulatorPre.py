from helpers.metrics import compute_napfd

test_result_encoding = {
    'PASSED': 0,
    'NO_STATUS': 0,
    'NO STATUS': 0,
    'FAILED_TO_BUILD': 0,
    'SKIPPED': 0,
    'FAILED TO BUILD': 0,
    'TIMEOUT': 0,
    'FLAKY': 0,
    'FAILED': 1
}

def encode_observation(observation, features, history_length):
    """
    Converts dataframe of job observation to an encoded dataframe for training.

    Parameters:
    observation (pd.DataFrame): DataFrame of job observations.
    features (list): List of features to consider.
    history_length (int): Length of the history to encode.

    Returns:
    pd.DataFrame: Encoded dataframe.
    """
    # Ordinal encoding for test results
    encoding = {"prev_test_result": test_result_encoding}
    for i in range(history_length):
        encoding[f'result_t-{i + 1}'] = test_result_encoding

    observation = observation.replace(encoding)

    return observation[features]

class CISimulatorPre:
    timestamp = 0
    observation = {}
    available_resources = 0.3
    database = None

    def __init__(self, db, evaluator, pipeline='POST', pca_components=9):
        """
        CI Simulator for check processing.

        Parameters:
        db (DatabaseBaseMock): Database object.
        evaluator (Evaluator): Evaluator object.
        pipeline (str): Pipeline name. Defaults to 'POST'.
        """
        self.database = db
        self.evaluator = evaluator
        self.pipeline = pipeline

        # If available, all these features are used as input features together with the historical test_results
        all_features = [
            't_last_execution', 'test_name_id'
        ]
        all_features += [f"PCA_{i}" for i in range(pca_components)]
        self.features = [feature for feature in self.database.get_feature_names() if feature in all_features]
        self.features += [f'result_t-{i + 1}' for i in range(db.history)]
        self.features += ['last_execution', 'last_failure', 'test_duration']

    def reset(self, time=None):
        """
        Reset CISimulator environment to a certain timestamp.
        Loads the job data of the timestamp, filters, and encodes the data.

        Parameters:
        time (int, optional): Timestamp to reset to. Defaults to None.

        Returns:
        pd.DataFrame: Current observation of the job.
        """
        self.timestamp = self.timestamp + 1 if time is None else time

        # Load observation for the specified time
        self.observation = self.database.get_job(time)
        if len(self.observation) == 0:
            return self.observation

        return encode_observation(self.observation, self.features, self.database.history)

    def step(self, action):
        """
        Perform one step in the CISimulator environment.

        Parameters:
        action (list): List of predictions. The higher, the likelier the test is to be executed.

        Returns:
        tuple: DataFrame containing all results such as rewards and further debug info, and a boolean indicating if the simulation is done.
        """
        # Schedule only the top X% of tests
        self.observation['prediction'] = action
        self.observation = self.observation.sort_values(by=['prediction'], ascending=False)
        self.observation['scheduled'] = False
        time_budget = self.observation['avg_test_duration'].sum() * self.available_resources
        scheduled_idxs = []

        for idx in list(self.observation.index):
            if self.observation.loc[idx, 'test_duration'] < time_budget:
                time_budget -= self.observation.loc[idx, 'test_duration']
                scheduled_idxs.append(idx)
            else:
                break
        self.observation.loc[scheduled_idxs, 'scheduled'] = True

        # Compute reward and update job observation
        self.observation['reward'] = 0
        reward = self.database.execute_tests(list(self.observation[self.observation.scheduled].test_name))
        self.observation.loc[scheduled_idxs, 'reward'] = reward

        # Compute evaluation metrics
        metrics = {}
        metrics['failed'] = len(self.observation[self.observation.hypothetical_reward == 1]) > 0
        # At least one test failed
        if metrics['failed']:
            # Sort by predicted priority
            self.observation = self.observation.sort_values(by=['prediction', 'test_duration'], ascending=[False, True], ignore_index=True)
            # Compute metrics
            apfd_metrics = compute_napfd(self.observation, "hypothetical_reward")
            metrics['napfd'] = apfd_metrics['napfd']
            metrics['ttf'] = apfd_metrics['ttf']
            metrics['ttf_duration'] = apfd_metrics['ttf_duration']

        self.evaluator.log_job_results(self.timestamp, metrics)
        self.timestamp += 1
        done = self.database.set_timestamp(self.timestamp)

        return self.observation, done
