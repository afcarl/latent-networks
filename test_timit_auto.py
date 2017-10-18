import re
import os
import numpy
from time import sleep

def get_free_gpuid():
    lines = os.popen("nvidia-smi").read()
    print lines
    if len(lines) == 0:
        return None
    gpu_mems = [line for line in lines.split('\n') if
                re.search('([0-9]+?)MiB / ([0-9]+?)MiB', line) is not None]
    gpu_free = []
    for gpu_id, gpu_line in enumerate(gpu_mems):
	fields = re.search('([0-9]+?)MiB / ([0-9]+?)MiB', gpu_line)
        assert fields is not None
	used = int(fields.group(1))
        total = int(fields.group(2))
        if used <= 20:
            gpu_free.append(gpu_id)

    if len(gpu_free) > 0:
        return gpu_free[0]
    return None


seeds = ["42", "144", "13"]
tasks = ["--weight_aux_nll 0.009 --weight_aux_gen 0.009 --use_h_in_aux",
         "--weight_aux_nll 0.015 --weight_aux_gen 0.015 --use_h_in_aux",
	 "--weight_aux_nll 0.025 --weight_aux_gen 0.025 --use_h_in_aux"]
model_type = "model"
base_dir = "./"
compile_dir = "/scratch/alsordon/theano/compile"

all_commands = []
status = open("test_timit_auto.txt", "w")
for seed in seeds:
    for task in tasks:
        command = "python train_lstm_timit.py"
	command += " %s" % task
        command += " --philly_datadir %s" % base_dir
        command += " --seed %s > /dev/null 2>&1 &" % seed
        all_commands.append(command)

print("Dispatching %s commands" % len(all_commands))
print all_commands
while len(all_commands) > 0:
    gpu_id = None
    while gpu_id is None:
        sleep(30.)
        gpu_id = get_free_gpuid()

    print("Got free gpu: %s" % gpu_id)
    assert len(all_commands) > 0

    command = all_commands.pop()
    prefix = "THEANO_FLAGS=device=gpu%s,lib.cnmem=0.9,floatX=float32 " % (
            gpu_id)
    command = prefix + command
    os.system(command)
    print("Dispatching: %s" % command)
    status.write("Dispatching: %s\n" % command)
    status.write("Commands left : %s\n" % len(all_commands))
    status.flush()

status.close()

