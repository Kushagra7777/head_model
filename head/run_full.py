import os
import subprocess
import sys
from datetime import datetime
import shutil

def run_command(command):
    """Runs a shell command and waits for it to complete."""
    process = subprocess.Popen(command, shell=True)
    process.communicate()

def main(fold, input_dir, working_dir, result_dir):
    # Create a timestamp
    date = datetime.now().strftime('%y%m%d-%H%M%S')
    
    # Update input_dir with the fold
    input_dir = os.path.join(input_dir, fold)
    
    # Remove working_dir if it exists
    if os.path.exists(working_dir):
        shutil.rmtree(working_dir)
    
    # Create result_dir if it does not exist
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # Run training jobs in parallel
    processes = []
    for i in range(5):
        command = f"python full.py --gpu=0 --input_dir={input_dir} --working_dir={os.path.join(working_dir, str(i))} --seeds={i} > {os.path.join(result_dir, f'train_log_{i}.txt')} 2>&1"
        processes.append(subprocess.Popen(command, shell=True))
    
    # Wait for all processes to complete
    for process in processes:
        process.wait()
    
    # Generate submission file
    command = f"python make_submission.py --mean=geometric --out {os.path.join(result_dir, 'submission.txt')} --out-probability {os.path.join(result_dir, 'probability.npy')} {os.path.join(working_dir, '*/*/lev2_xgboost2/test.h5')} > {os.path.join(result_dir, 'make_submission_log.txt')} 2>&1"
    run_command(command)
    
    # Evaluate the model
    command = f"python evaluate.py --prediction {os.path.join(result_dir, 'submission.txt')} --ground-truth {os.path.join(input_dir, 'test/labels.txt')} -l {os.path.join(input_dir, 'label_names.txt')} -o {result_dir} > {os.path.join(result_dir, 'evaluate_log.txt')} 2>&1"
    run_command(command)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python run_full.py <fold> <input_dir> <working_dir> <result_dir>")
        sys.exit(1)

    fold = sys.argv[1]
    input_dir = sys.argv[2]
    working_dir = sys.argv[3]
    result_dir = sys.argv[4]

    main(fold, input_dir, working_dir, result_dir)


    # python run_full.py 0 work_dir/0 work_dir/something work_dir/results_0

