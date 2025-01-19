def compute_information_gain_non_flaky(test_item, test_history):
    """
    Compute developer-relevant information gain for test execution.

    Parameters:
    test_item (pd.Series): Series containing the details of the current test execution.
    test_history (dict): Dictionary containing the historical data of the test.

    Returns:
    float: Information gain value.
    """
    if test_item['flaky']:
        return -1
    elif test_item['test_result'] == test_history['result_t-1']:
        return -test_item['test_duration']
    elif test_item['test_result'] == 'FAILED':
        information_gain_value = 1
    elif test_item['test_result'] == 'PASSED' and test_history['result_t-1'] == 'FAILED':
        information_gain_value = 1
    else:
        information_gain_value = -test_item['test_duration']

    return information_gain_value
