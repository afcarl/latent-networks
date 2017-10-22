
import numpy as np
import glob as glob
import sys
import os

def get_seed(filename):
    fz = filename.split('_')
    return fz[0], '_'.join(fz[1:])

def get_perf(filename):
    nepochs = 0.
    perf = []
    with open(filename, 'r') as f:
        for line in f:
            if line.find('Valid') == 0:
                line = line.split(':')
                valid, test = line[1].split(',')
                nepochs += 1
                perf.append((float(valid), float(test)))
    return perf

log_files = glob.glob(sys.argv[1] + "*.txt")
results = {}
for log_file in log_files:
    exp_name = os.path.basename(log_file)
    seed, name = get_seed(exp_name)
    results[name] = results.get(name, []) + [get_perf(log_file)[-1]]

for name, res in results.items():
    valid, test = zip(*res)    
    print(name, np.mean(valid), np.mean(test))
