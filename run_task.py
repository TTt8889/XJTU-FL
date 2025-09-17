import subprocess
import re
import os

def execute_command(command):
    try:
        # Use Popen to start the process with shell=True
        process = subprocess.Popen(command, shell=True)
        print(f"Starting process: {command}")
        print()
        return process
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_output_filename(command, log_directory):
    """
    Dynamically generate the output filename based on the parameters in the command.
    """
    # Extract the dataset name from the -config parameter
    config_match = re.search(r"-config=\.\/configs\/my_exp_(\w+).yaml", command)
    dataset_name = config_match.group(1) if config_match else "unknown_dataset"

    # Extract the npy file name
    npy_match = re.search(r"--npy_file_name=([^\s]+)", command)
    npy_file_name = npy_match.group(1) if npy_match else "unknown_npy"

    # Extract the attack and defense methods
    attack_match = re.search(r"--attack=(\w+)", command)
    defense_match = re.search(r"--defense=(\w+)", command)
    attack_method = attack_match.group(1) if attack_match else "unknown_attack"
    defense_method = defense_match.group(1) if defense_match else "unknown_defense"

    # Generate the output filename
    output_filename = f"{dataset_name}_{npy_file_name}_{attack_method}_{defense_method}.txt"
    return os.path.join(log_directory, output_filename)

if __name__ == "__main__":
    # List of original shell commands

    # python main.py -config=./configs/my_exp_MNIST.yaml --gpu_idx=xx --npy_file_name=xx --attack=xx --defense=xx     
    # python main.py -config=./configs/my_exp_CIFAR10.yaml --gpu_idx=xx --npy_file_name=xx --attack=xx --defense=xx
    
    shell_commands = [
        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=1 --npy_file_name=3.3.2noiid --distribution=non-iid --dirichlet_alpha=0.8 "
        #     "--attack=NoAttack --defense=Mean"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=1 --npy_file_name=3.3.2noiid --distribution=non-iid --dirichlet_alpha=0.8 "
        #     "--attack=NoAttack --defense=TrimmedMean"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=1 --npy_file_name=3.3.2noiid --distribution=non-iid --dirichlet_alpha=0.8 "
        #     "--attack=NoAttack --defense=NormClipping"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=1 --npy_file_name=3.3.2noiid --distribution=non-iid --dirichlet_alpha=0.8 "
        #     "--attack=NoAttack --defense=CenteredClipping"),

        # (   "python main.py -config=./configs/my_exp_MNIST.yaml "
        #     "--gpu_idx=2 --npy_file_name=3.3.2noiid --distribution=non-iid --dirichlet_alpha=0.8 "
        #     "--attack=NoAttack --defense=Mean"),

        # (   "python main.py -config=./configs/my_exp_MNIST.yaml "
        #     "--gpu_idx=2 --npy_file_name=3.3.2noiid --distribution=non-iid --dirichlet_alpha=0.8 "
        #     "--attack=NoAttack --defense=TrimmedMean"),

        # (   "python main.py -config=./configs/my_exp_MNIST.yaml "
        #     "--gpu_idx=2 --npy_file_name=3.3.2noiid --distribution=non-iid --dirichlet_alpha=0.8 "
        #     "--attack=NoAttack --defense=NormClipping"),

        # (   "python main.py -config=./configs/my_exp_MNIST.yaml "
        #     "--gpu_idx=2 --npy_file_name=3.3.2noiid --distribution=non-iid --dirichlet_alpha=0.8 "
        #     "--attack=NoAttack --defense=CenteredClipping"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2noiid --distribution=non-iid --dirichlet_alpha=0.3 "
        #     "--num_adv=0.3 --attack=MyAttack --defense=Mean"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2noiid --distribution=non-iid --dirichlet_alpha=0.3 "
        #     "--num_adv=0.3 --attack=MyAttack --defense=TrimmedMean"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2noiid --distribution=non-iid --dirichlet_alpha=0.3 "
        #     "--num_adv=0.3 --attack=MyAttack --defense=NormClipping"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2noiid --distribution=non-iid --dirichlet_alpha=0.3 "
        #     "--num_adv=0.3 --attack=MyAttack --defense=CenteredClipping"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=1 --npy_file_name=3.3.2hyperE --num_adv=0.2 "
        #     "--attack=MyAttack --defense=NormClipping"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=1 --npy_file_name=3.3.2hyperE --num_adv=0.2 "
        #     "--attack=MyAttack --defense=CenteredClipping"),

        # (   "python main.py -config=./configs/my_exp_MNIST.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2 --num_adv=0.2 "
        #     "--attack=FangAttack --defense=CenteredClipping"),

        # (   "python main.py -config=./configs/my_exp_MNIST.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2 --num_adv=0.2 "
        #     "--attack=SignFlipping --defense=CenteredClipping"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=2 --npy_file_name=3.3.2 --num_adv=0.3 "
        #     "--attack=MyAttack --defense=Mean"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=2 --npy_file_name=3.3.2 --num_adv=0.3 "
        #     "--attack=MyAttack --defense=Krum"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=2 --npy_file_name=3.3.2 --num_adv=0.3 "
        #     "--attack=MyAttack --defense=MultiKrum"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2 --num_adv=0.2 "
        #     "--attack=NoAttack --defense=CenteredClipping"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2 --num_adv=0.2 "
        #     "--attack=ALIE --defense=CenteredClipping"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2 --num_adv=0.2 "
        #     "--attack=IPM --defense=CenteredClipping"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2 --num_adv=0.2 "
        #     "--attack=MinMax --defense=CenteredClipping"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2 --num_adv=0.2 "
        #     "--attack=MinSum --defense=CenteredClipping"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2 --num_adv=0.2 "
        #     "--attack=SignFlipping --defense=CenteredClipping"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2 --num_adv=0.2 "
        #     "--attack=FangAttack --defense=CenteredClipping"),

        # (   "python main.py -config=./configs/my_exp_CIFAR10.yaml "
        #     "--gpu_idx=3 --npy_file_name=3.3.2 --num_adv=0.2 "
        #     "--attack=Gaussian --defense=CenteredClipping"),

        # (         
        #     "python main.py -config=./configs/my_exp_CIFAR10.yaml "         
        #     "--gpu_idx=3 --npy_file_name=4.3.2 "         
        #     "--attack=SignFlipping --defense=Krum"     
        # ),     

        # (         
        #     "python main.py -config=./configs/my_exp_CIFAR10.yaml "         
        #     "--gpu_idx=3 --npy_file_name=4.3.2 "         
        #     "--attack=SignFlipping --defense=MultiKrum"     
        # ),

        # (         
        #     "python main.py -config=./configs/my_exp_CIFAR10.yaml "         
        #     "--gpu_idx=3 --npy_file_name=4.3.2 "         
        #     "--attack=SignFlipping --defense=TrimmedMean"     
        # ),

        # (         
        #     "python main.py -config=./configs/my_exp_CIFAR10.yaml "         
        #     "--gpu_idx=3 --npy_file_name=4.3.2 "         
        #     "--attack=SignFlipping --defense=Median"     
        # ),

        # (         
        #     "python main.py -config=./configs/my_exp_CIFAR10.yaml "         
        #     "--gpu_idx=3 --npy_file_name=4.3.2 "         
        #     "--attack=SignFlipping --defense=Bulyan"     
        # ),

        # (         
        #     "python main.py -config=./configs/my_exp_CIFAR10.yaml "         
        #     "--gpu_idx=3 --npy_file_name=4.3.2 "         
        #     "--attack=SignFlipping --defense=NormClipping"     
        # ),

        # (         
        #     "python main.py -config=./configs/my_exp_MNIST.yaml "         
        #     "--gpu_idx=0 --npy_file_name=4.3.2duibi --num_adv=0.3 "         
        #     "--attack=MyAttack --defense=MyDefense"     
        # ),

        # (         
        #     "python main.py -config=./configs/my_exp_MNIST.yaml "         
        #     "--gpu_idx=2 --npy_file_name=4.3.2rs --softmax_seg=Normal "         
        #     "--attack=MinMax --defense=MyDefense"     
        # ),

        # (         
        #     "python main.py -config=./configs/my_exp_CIFAR10.yaml "         
        #     "--gpu_idx=1 --npy_file_name=4.3.2duibi --num_adv=0.3 "         
        #     "--attack=MyAttack --defense=MyDefense"     
        # ),

        # (         
        #     "python main.py -config=./configs/my_exp_CIFAR10.yaml "         
        #     "--gpu_idx=3 --npy_file_name=4.3.2rs --softmax_seg=Normal "         
        #     "--attack=MinMax --defense=MyDefense"     
        # ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "         
            "--gpu_idx=3 --npy_file_name=3.1.1 "         
            "--attack=MyAttack --defense=NormClipping"     
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "         
            "--gpu_idx=0 --npy_file_name=3.1.1 "
            "--attack=ALIE --defense=NormClipping"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=1 --npy_file_name=3.1.1 "
            "--attack=IPM --defense=NormClipping"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=1 --npy_file_name=3.1.1 "
            "--attack=Gaussian --defense=NormClipping"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=2 --npy_file_name=3.1.1 "
            "--attack=FangAttack --defense=NormClipping"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=2 --npy_file_name=3.1.1 "
            "--attack=MPAF --defense=NormClipping"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=0 --npy_file_name=3.1.1 "
            "--attack=NoAttack --defense=NormClipping"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "         
            "--gpu_idx=3 --npy_file_name=3.1.1 "         
            "--attack=MyAttack --defense=TESSERACT"     
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "         
            "--gpu_idx=0 --npy_file_name=3.1.1 "
            "--attack=ALIE --defense=TESSERACT"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=1 --npy_file_name=3.1.1 "
            "--attack=IPM --defense=TESSERACT"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=1 --npy_file_name=3.1.1 "
            "--attack=Gaussian --defense=TESSERACT"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=2 --npy_file_name=3.1.1 "
            "--attack=FangAttack --defense=TESSERACT"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=2 --npy_file_name=3.1.1 "
            "--attack=MPAF --defense=TESSERACT"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=0 --npy_file_name=3.1.1 "
            "--attack=NoAttack --defense=TESSERACT"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "         
            "--gpu_idx=3 --npy_file_name=3.1.1 "         
            "--attack=MyAttack --defense=CenteredClipping"     
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "         
            "--gpu_idx=0 --npy_file_name=3.1.1 "
            "--attack=ALIE --defense=CenteredClipping"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=1 --npy_file_name=3.1.1 "
            "--attack=IPM --defense=CenteredClipping"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=1 --npy_file_name=3.1.1 "
            "--attack=Gaussian --defense=CenteredClipping"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=2 --npy_file_name=3.1.1 "
            "--attack=FangAttack --defense=CenteredClipping"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=2 --npy_file_name=3.1.1 "
            "--attack=MPAF --defense=CenteredClipping"
        ),

        (         
            "python main.py -config=./configs/my_exp_MNIST.yaml "
            "--gpu_idx=0 --npy_file_name=3.1.1 "
            "--attack=NoAttack --defense=CenteredClipping"
        ),
    ]

    # Directory to store log files
    log_directory = "ZTask_Log"
    os.makedirs(log_directory, exist_ok=True)  # Create the log directory if it doesn't exist

    # Check if the directory exists and is a valid directory
    if os.path.exists(log_directory) and os.path.isdir(log_directory):
        # Iterate through all files and subdirectories in the directory
        for filename in os.listdir(log_directory):
            # Construct the full path of the file
            file_path = os.path.join(log_directory, filename)
            
            # Check if it is a file and not "runtask.txt"
            if os.path.isfile(file_path) and filename != "runtask.txt":
                # Delete the file
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
    else:
        print(f"The directory {log_directory} does not exist or is not a valid directory.")


    # Convert commands to background execution format and store in a new list
    background_commands = []
    for command in shell_commands:
        output_file = generate_output_filename(command, log_directory)  # Dynamically generate the output filename
        background_command = f"nohup {command} >> {output_file} 2>&1 &"
        background_commands.append(background_command)

    # Start all background commands
    for command in background_commands:
        execute_command(command)

    print("All processes have been started. The main program continues to execute.")


# 注意运行前修改好yaml配置文件
# nohup python run_task.py >> ZTask_Log/runtask.txt 2>&1 &