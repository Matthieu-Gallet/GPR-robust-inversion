'''
File: launch_experiment.py
Created Date: Mon Jan 09 2023
Author: Ammar Mian
-----
Last Modified: Mon Jan 16 2023
Modified By: Ammar Mian
-----
Copyright (c) 2023 Université Savoie Mont-Blanc
-----
File to manage the launch of an experiment.
'''

import argparse
import yaml
from datetime import datetime
import os
import shutil
import logging
from subprocess import call, run
import git
from rich import print as rprint
import stat

def create_results_directory(
    experiment_file, results_dir, runner,
    commit_sha, is_repo_dirty):
    
    # Creating directory if it doesn't exists
    now = datetime.now() # current date and time
    date_time = now.strftime("%m-%d-%Y:%H-%M-%S")
    experiment_dir_name = os.path.basename(experiment_file).split('.')[0]
    directory_path = os.path.join(results_dir, experiment_dir_name+"_"+date_time)
    print(f"Creating results directory: {directory_path}")
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    else:
        print(f"Error: experiment folder {directory_path}  already exists, finishing now")
        exit(0)
        
    # Copying configuration file to the new directory
    shutil.copy(experiment_file, os.path.join(directory_path, "experiment.yml"))

    
    # Creating log files and redirecting outputs and error
    logging.basicConfig(filename=os.path.join(directory_path, "experiment_management.log"), 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        filemode='w') 

    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(f"Directory for experiment \"{experiment_dir_name}\" has been initialized!")

    # Creating experiment running metadata file
    running_metadata = {
        "runner": runner,
        "start_date": date_time,
        "end_date": None,
        "commit_sha": commit_sha,
        "experiment_ended":  False,
        "metric": "",
        "is_repo_dirty": is_repo_dirty
    }
    with open(os.path.join(directory_path, "running_metadata.yml"), 'w') as f:
        yaml.dump(running_metadata, f)
    return logger, directory_path


if __name__ == "__main__":
    
    # Adding arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_file",  help="YAML experiment configuration file")
    parser.add_argument("--runner",  default="local", choices=["local", "job"],
        help="Choice between local(=running on the local machine) or job (=as a job through HTcondor system installed on the machine)"
    )
    parser.add_argument("--results_dir", default="./results", help="Location of results directory for artifact and logs storage")
    parser.add_argument("--submit_template", help="HTcondor submit template for ressources to take when running as a job.")
    parser.add_argument('--force_without_commit', action='store_true', 
        default=False, help="Force execution without commiting the reportory first. Not recommended for production"
    )
    args = parser.parse_args()


    # Making Submit template mandatory for a job execution
    if args.runner=="job" and (args.submit_template is None):
        parser.error(f"It is mandatory to specify a HTCondor submit template for execution with a job type execution!")

    # Checking if git repo is clean, i.e all changes have been committed
    repo = git.Repo('.')
    if not args.force_without_commit:
        try:
            assert not repo.is_dirty()
        except AssertionError:
            rprint("[bold red]The repertory has not been committed. Please commit before launching an experiment ![/bold red]")
            rprint("[green]It is possible to run with option --force_without_commit. But it is not recommended if some models or code has been changed.")
            exit(0)

        rprint("[green]Git repertory is clean. Proceeding with experiment.[/green]")
    else:
        rprint("[bold red]Launching experiment without committing. The last commit will be used as a reference.[/bold red]")

    commit_sha = repo.head.object.hexsha


    # Creating resulting directory now
    logger, directory_path = create_results_directory(
        args.experiment_file, args.results_dir, args.runner,
        commit_sha, repo.is_dirty()
    )
    logger.info("Launching experiment.")
    logger.info(f"Git commit sha: {commit_sha}")
    
    execute_command = f"python execute_experiment.py {args.experiment_file} {directory_path}"
    # Managing an execution on the local machine versus submitting a job through HTcondor
    if args.runner == "local":
        print("Executing on the local machine")
        logger.info("Executing on the local machine")

        logger.info(f"Now running command:\n {execute_command}")
        call(["python", "execute_experiment.py", f"{args.experiment_file}", f"{directory_path}"])


        # Handle end of experiment by putting a flag in the running_metadata_file
        # It allows to handle difference between job not done and job had an error
        with open(os.path.join(directory_path, "running_metadata.yml"),'r') as f:
            running_metadata = yaml.load(f, Loader=yaml.FullLoader)
        running_metadata['experiment_ended'] = True
        with open(os.path.join(directory_path, "running_metadata.yml"), 'w') as f:
            yaml.dump(running_metadata, f)
        logger.info("Experiment done. Quitting.")


    # Submitting a job through HTcondor
    else:
        print("Submitting job through HTCondor")

        # Creating a submit file based upon template
        shutil.copy(args.submit_template, os.path.join(directory_path, "job.submit"))
        with open(os.path.join(directory_path, "job.submit"), 'a') as f:
            f.write(f"executable={os.path.join(directory_path, 'job.sh')}\n")
            f.write(f"log={os.path.join(directory_path, 'job.log')}\n")
            f.write(f"queue")


        # Creating bash file to run command and tell experiment ended
        with open(os.path.join(directory_path, "job.sh"), 'w') as f:
            f.write("#!/usr/bin/bash\n")
            f.write(f"cd {os.getcwd()}\n")
            f.write(execute_command+"\n")
            f.write(f"sed -i \"s/experiment_ended: false/experiment_ended: true/\" {os.path.join(directory_path, 'running_metadata.yml')}\n")
            f.write(f"echo \"Job Done. Quitting.\"")


        # Make job.sh executable
        st = os.stat(os.path.join(directory_path, "job.sh"))
        os.chmod(os.path.join(directory_path, "job.sh"), st.st_mode | stat.S_IEXEC)

        # # Submit job to the cluster
        run(["condor_submit" ,f"{os.path.join(directory_path, 'job.submit')}"])
