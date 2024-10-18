import os
import time

# wait for process to finish for all
def wait_file(files,time_to_wait = 365.25 * 24 * 60 * 60,refresh=30):
    time_counter = 0
    while not all(list(map(os.path.isfile, files))):
        time.sleep(refresh)
        time_counter += refresh
        if time_counter > time_to_wait:
            print(f"Too bad, time's up!")
            break
    print(f'Done after {time_counter}!')
