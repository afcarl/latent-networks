import time

old_check = time.time()

def print_philly_hb(each_n_minutes=5):
    global old_check
    cur_check = time.time()
    if (cur_check - old_check) >= 60 * each_n_minutes:
        print('PROGRESS: 00.00%')
        old_check = time.time()
    return
