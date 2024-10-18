import os
import subprocess as s
import time
import re

def call_script(dirnam, stdout=None, **kwargs):

    # save current dir and move to dirnam
    cwd = os.getcwd()
    os.chdir(dirnam)

    if 'is_slurm' in kwargs:
        if kwargs['is_slurm']:
            to_call = ['sbatch']+list(kwargs.values())
        else:
            to_call = list(kwargs.values())
    else:
        to_call = list(kwargs.values())

    flat_list = []
    _ = [flat_list.extend(item) if isinstance(item, list) else flat_list.append(item) for item in to_call if item]

    if stdout != None:
        with open(stdout, "w+") as outfile:
            iostat = s.call(flat_list, stdout=outfile)
    else:
        iostat = s.call(flat_list)  # does not wait, add following lines to wait
        
    # go back to original dir
    os.chdir(cwd)

def check_output_script(dirnam, timeout=None, **kwargs):

    # save current dir and move to dirnam
    cwd = os.getcwd()
    os.chdir(dirnam)

    to_call = list(kwargs.values())
    flat_list = []
    _ = [flat_list.extend(item) if isinstance(item, list) else flat_list.append(item) for item in to_call if item]
    #print(flat_list)
    output = s.check_output(flat_list, timeout)

    # go back to original dir
    os.chdir(cwd)

    return output


def popen_script(dirnam, pattern, timeout=300, delay=1, **kwargs):
    """
    Executes a script in the specified directory and waits until a given pattern is found in the output.

    Args:
    - dirnam: Directory in which to execute the script.
    - pattern: The pattern to wait for in the output.
    - timeout: Maximum time (in seconds) to wait for the pattern.
    - delay: Delay (in seconds) between each check of the output.
    - kwargs: Additional arguments for the script.

    Returns:
    - output: The output of the script when the pattern is found.
    """
    # Save current dir and move to dirnam
    cwd = os.getcwd()
    os.chdir(dirnam)

    to_call = list(kwargs.values())
    flat_list = []
    _ = [flat_list.extend(item) if isinstance(item, list) else flat_list.append(item) for item in to_call if item]

    process = s.Popen(flat_list, stdout=s.PIPE, stderr=s.PIPE, text=True)
    output = ""
    pattern_found = False
    start_time = time.time()

    while time.time() - start_time < timeout:
        line = process.stdout.readline()
        if not line:
            time.sleep(delay)
            continue
        output += line
        if re.search(pattern, line):
            pattern_found = True
            break

    # Terminate the process if the pattern is not found within the timeout
    process.terminate()
    process.wait()

    # Go back to original dir
    os.chdir(cwd)

    if not pattern_found:
        raise TimeoutError(f"Pattern '{pattern}' not found in output within {timeout} seconds")

    return output
