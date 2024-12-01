import os

# Define your variables
input_dir = "work_dir/0"  # Replace with your actual input directory path
working_dir = "work_dir/something"  # Replace with your actual working directory path
result_dir = "work_dir/results_0"  # Replace with your actual result directory path

# Ensure the result directory exists
os.makedirs(result_dir, exist_ok=True)

# Run the script sequentially for each value of i
for i in range(5):
    # Define the command string
    command = (
        f"python full.py "
        f"--gpu=0 "
        f"--input_dir={input_dir} "
        f"--working_dir={working_dir}/{i} "
        f"--seeds={i} "
        f"> {os.path.join(result_dir, f'train_log_{i}.txt')}"
    )
    
    # Print the command for debugging
    print(f"Running command: {command}")
    
    # Run the command and wait for it to complete
    exit_code = os.system(command)
    
    if exit_code != 0:
        print(f"Process {i} failed with exit code {exit_code}")
    else:
        print(f"Process {i} completed successfully.")


# python3 full.py --gpu=0 --input_dir=work_dir/0 --working_dir=work_dir/something/0 --seeds=0
