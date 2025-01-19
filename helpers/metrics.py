import numpy as np

def compute_napfd(df, hypothetical_reward_name="hypothetical_information_gain"):
    """
    Compute the Normalized Average Percentage of Faults Detected (NAPFD) metric.

    Parameters:
    df (pd.DataFrame): DataFrame containing test results and related information.
    hypothetical_reward_name (str): Column name for the hypothetical reward. Defaults to "hypothetical_information_gain".

    Returns:
    dict: Dictionary containing various computed metrics.
    """
    detection_ranks = []

    sorted_df = df.sort_values(by=["prediction"], ascending=False)
    for idx in range(len(sorted_df)):
        if sorted_df.iloc[idx].scheduled and sorted_df.iloc[idx].reward > 0:
            detection_ranks.append(idx + 1)

    costs = df["test_duration"].to_list()
    detection_cost = 0
    for i in detection_ranks:
        detection_cost += sum(costs[i - 1:]) - 0.5 * costs[i - 1]

    detected_changes = len(df[(df.scheduled) & (df.reward > 0)])
    total_changes_count = len(df[df[hypothetical_reward_name] > 0])
    undetected_changes = len(df[(df.scheduled == False) & (df[hypothetical_reward_name] > 0)])
    nr_testcases = len(df)

    ttf_duration = np.nan
    ttf = np.nan
    execution_delay = []
    last_execution = []
    duration_scheduled = list(df[df.scheduled].test_duration)
    duration_non_scheduled = list(df[df.scheduled == False].test_duration)

    if "execution_delay" in df:
        execution_delay = list(df[(df.scheduled) & (df.reward > 0)].execution_delay)
        last_execution = list(df[df.last_execution >= 0].last_execution)

    if detected_changes > 0:
        first_failed_test = ((sorted_df[hypothetical_reward_name] == 1) & (sorted_df.scheduled)).argmax() + 1
        ttf_duration = sorted_df[sorted_df.scheduled].test_duration[:first_failed_test].sum()
        cycle_duration = sorted_df[sorted_df.scheduled].test_duration.sum()
        ttf_duration /= cycle_duration
        ttf = first_failed_test / len(sorted_df[sorted_df.scheduled])

    if len(df) <= 6:
        return {
            "napfd": np.nan,
            "recall": np.nan,
            "apfdc": np.nan,
            "apfd": np.nan,
            "ttf": ttf,
            "ttf_duration": ttf_duration,
            "p": np.nan,
            "last_execution": last_execution,
            "execution_delay": execution_delay,
            "duration_scheduled": duration_scheduled,
            "duration_non_scheduled": duration_non_scheduled,
        }

    assert detected_changes + undetected_changes == total_changes_count

    if total_changes_count > 0:
        p = detected_changes / total_changes_count if undetected_changes > 0 else 1
        napfd = (
            p - sum(detection_ranks) / (total_changes_count * nr_testcases) + p / (2 * nr_testcases)
        )

    return {
        "napfd": napfd,
        "ttf": ttf,
        "ttf_duration": ttf_duration,
        "p": p,
        "last_execution": last_execution,
        "execution_delay": execution_delay,
        "duration_scheduled": duration_scheduled,
        "duration_non_scheduled": duration_non_scheduled,
    }
