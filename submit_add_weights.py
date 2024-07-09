import subprocess
import os
import sys

def run_sample_weight_add(toml):
    print(toml)
    try:
        script = '/home/hpc/capn/mppi132h/scripts/data_set_creation/sample_weigths/calc_single_weights.py'
        command = 'python '+script+' '+toml
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Return code: {e.returncode}")
    else:
        print(f"Command executed successfully: {command}")



if __name__ == '__main__':
    test_toml = r"C:\Users\basti\PythonScripts\Master_thesis\test_input.toml"
    run_sample_weight_add(test_toml)