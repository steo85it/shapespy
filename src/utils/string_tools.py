import re

def extract_thresholds(log_data, pattern):
    """
    log_data: list of strings
    @type pattern: Regular expression to match image paths and their corresponding threshold values
    """
    # Decode bytes to string if necessary
    if isinstance(log_data, bytes):
        log_data = log_data.decode('utf-8')

    # Find all matches in the log data
    matches = re.findall(pattern, log_data)

    # Create a dictionary to store the results
    thresholds = {match[0]: float(match[1]) for match in matches}

    return thresholds

