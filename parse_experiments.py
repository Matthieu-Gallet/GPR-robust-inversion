import os, sys
from simple_term_menu import TerminalMenu
from rich import print as rprint
from rich.console import Console
from rich.table import Table
import yaml
from datetime import datetime
import hashlib
import pandas as pd
import pydoc
from subprocess import call, Popen, PIPE, STDOUT
import argparse

def main_menu():
    """Main menu of the parsing.

    Returns
    -------
    int
        A choice between all the options
    """
    choices = [
        "[a] show all experiments",
        "[b] filter experiments with tags",
        "[c] filter experiments by configuration",
        "[d] look up information on experiment",
        "[q] quit"]
    terminal_menu = TerminalMenu(choices, title="GPR inversion experiments explore menu")
    menu_entry_index = terminal_menu.show()
    return menu_entry_index

def get_all_experiments_directories(results_dir):
    """Get all the experiments directory names for this project.

    Parameters
    ----------
    results_dir: str
        path to results storing dir

    Returns
    -------
    list
        list of str containing directory names
    """
    return [name for name in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, name))]

def parse_one_experiment(directory_name, results_dir):
    """Parse one experiment data from its directory name

    Parameters
    ----------
    directory_name : str
        directory name, not path
    results_dir: str
        path to results storing dir

    Returns
    -------
    tuple
        parsed data about the experiment
    """
    path = os.path.join(results_dir, directory_name)


    # Get date from running metdata file
    try:
        with open(os.path.join(path, "running_metadata.yml"), 'r')  as f:
            running_metada = yaml.load(f, Loader=yaml.FullLoader)
        begin_time = datetime.strptime(running_metada['start_date'], "%m-%d-%Y:%H-%M-%S")
    except  FileNotFoundError:
        rprint(f"[bold red]Experiment files are not up to format for {path}[/bold red]. Dropping it.")
        return -1


    #  TODO: when we kill a job, there is no experiment_ended update so it appears as running
    # Case experiment is done running
    if running_metada['end_date'] is not None:
        finish_time = datetime.strptime(running_metada['end_date'], "%m-%d-%Y:%H-%M-%S")
        duration = str(finish_time - begin_time).split('.')[0]
        status = u'\u2713'
    # The experiment is not done or had an error
    elif not running_metada["experiment_ended"]: 
        status= "\u21BB"
        finish_time = datetime.now()
        duration = str(finish_time - begin_time).split('.')[0]
    else: # There was an error so the end_time wasn't put
        status = u'\u2717'
        duration = ""


    # parse experiment configuration file
    experiment_file = os.path.join(path, "experiment.yml")
    with open(experiment_file, 'rb')  as f:
        experiment = yaml.load(f, Loader=yaml.FullLoader)

    return (
        running_metada['start_date'], experiment['name'], ", ".join(experiment['tags']),
        status, duration, running_metada["metric"], running_metada["commit_sha"],
        os.path.join(results_dir, directory_name)
    )


def get_id_from_experiment_directory(directory_name):
    """Hash a directory name to produce an unique id to refer to the experiment.

    Parameters
    ----------
    directory_name : str
        name of the directory saving the experiment. With the date and everything

    Returns
    -------
    str
        hashed id
    """
    return hashlib.md5(directory_name.encode("utf-8")).hexdigest()


def parse_all_experiments(results_dir):
    """Parse all directories in results to get the data from all experiments.
    TODO: Improve to handle different directories.

    Returns
    -------
    DataFrame
        Pandas DataFrame storing all info.
    """
    list_experiments = get_all_experiments_directories(results_dir)
    table = []
    for experiment in list_experiments:
        experiment_data = parse_one_experiment(experiment, results_dir)
        if experiment_data != -1:
            id = get_id_from_experiment_directory(experiment)
            data = [id] + list(experiment_data)
            table.append(data)
    df = pd.DataFrame(table, columns=["ID", "Start date", "Name", "Tags", "Status", "Duration", "Metric", "Commit Sha", "Path"])
    return df.sort_values("Start date", ascending=False)


def menu_select_one_experiment():
    """Menu for a single experiment lookup. Search by ID aor selection.

    Returns
    -------
    int
        choice
    """
    choices = [
        "[a] look up by id",
        "[b] select from experiments",
        "[q] cancel"]
    terminal_menu = TerminalMenu(choices, title="Lookup an experiment")
    menu_entry_index = terminal_menu.show()
    return menu_entry_index


def show_data_one_experiment(expriment_id, parsing_data):
    """Show configuration data + ask if wanting to see logs and/or plots.
    DO NOT CHECK IF ID EXISTS OR NOT.

    Parameters
    ----------
    expriment_id : str
        id of experiment
    parsing_data : DataFrame
        parsed data from all experiments
    """

    # Get row of experiment
    line = parsing_data.loc[parsing_data["ID"]==expriment_id]
    directory_path = line.to_numpy().flatten()[-1]
    rprint(f"[bold]Showing information for experiment {expriment_id} located at:\n{directory_path}\n")

    # Action menu on this specific experiment
    choices = [
        "[a] Show experiment configuration file",
        "[b] Show experiment management log", 
        "[c] Show experiment execution log",
        "[d] Show experiment running metadata"]
    
    if os.path.exists(os.path.join(directory_path, "job.submit")):
        choices.append('[e] Show job submit file')
        choices.append('[f] Show job log')

    if os.path.exists('plot_experiment.py') and line["Status"].to_list()[0] == u'\u2713':
        choices.append("[p] Plot experiment")
    choices.append("[q] quit")

    terminal_menu = TerminalMenu(choices, title="Select an option")
    menu_entry_index = -1
    while menu_entry_index != len(choices)-1:
        terminal_menu = TerminalMenu(choices, title="Select an option")
        menu_entry_index = terminal_menu.show()

        if menu_entry_index ==  0:
            # Read the yaml file to print configuration
            yaml_path = os.path.join(directory_path, "experiment.yml")
            with open(yaml_path, 'r') as f:
                pydoc.pager(f.read())

        # Option : show experiment management log
        elif menu_entry_index == 1:
            experiment_management_file = os.path.join(directory_path, "experiment_management.log")
            with open(experiment_management_file, 'r') as f:
                pydoc.pager(f.read())

        # Option : show experiment management log
        elif menu_entry_index == 2:
            experiment_execution_file = os.path.join(directory_path, "experiment_execution.log")
            with open(experiment_execution_file, 'r') as f:
                pydoc.pager(f.read())

        # Show running metadata
        elif menu_entry_index == 3:
            # Read the yaml file to print metadata
            yaml_path = os.path.join(directory_path, "running_metadata.yml")
            with open(yaml_path, 'r') as f:
                pydoc.pager(f.read())

        # Option : show submit file only if option is available
        elif menu_entry_index == 4 and choices[4] == "[e] Show job submit file":
            submit_file = os.path.join(directory_path, "job.submit")
            with open(submit_file, 'r') as f:
                pydoc.pager(f.read())

        # Option : show submit file only if option is available
        elif menu_entry_index == 5 and choices[5] == "[f] Show job log":
            log_file = os.path.join(directory_path, "job.log")
            with open(log_file, 'r') as f:
                pydoc.pager(f.read())

        # Option : show plots only if option is available
        elif menu_entry_index == len(choices)-2 and\
            choices[len(choices)-2] == "[p] Plot experiment":
            # Popen([sys.executable, f"python plot_experiment.py {directory_path}"], 
            #                         stdout=PIPE, 
            #                         stderr=STDOUT)
            call(["python", "plot_experiment.py", f"{directory_path}"])


def action_show_one_experiment(parsing_data):
    """Action for selection of "[c] look up information on experiment" in the main menu.

    Parameters
    ----------
    parsing_data : DataFrame
        parsed data from all experiments directories
    """
    choice = menu_select_one_experiment()
    if choice == 0:
        id = input("Enter the id of experiment: ")
        if id in parsing_data["ID"].values:
            show_data_one_experiment(id, parsing_data)
        else:
            rprint(f"[bold red]ID {id} not found in experiments![/bold red]")

    elif choice == 1:
        # TODO improve lookup in pandas so that <e don't have to lookup id.
        # For now: temporary solution because the sorting of pandas doesn't change
        # row number. thus the choice is wrong if we lookup indice menu
        choices = []
        ids = []
        for row in parsing_data.iterrows():
            choices.append(" // ".join(row[1].to_list()))
            ids.append( row[1]['ID'] )
        terminal_menu = TerminalMenu(choices, title="Select an experiment")
        menu_entry_index = terminal_menu.show()

        show_data_one_experiment(ids[menu_entry_index], parsing_data)

    return parsing_data


def action_show_all_experiments(parsing_data):
    """Action for selection of "[a] show all experiments" in the main menu.

    Parameters
    ----------
    parsing_data : DataFrame
        parsed data from all experiments directories
    """
    table = Table(title="List of experiments done")

    table.add_column("ID", justify="left", style="white", no_wrap=True)
    table.add_column("Start Date", justify="center", style="cyan", no_wrap=True)
    table.add_column("Name", justify="left", style="magenta", no_wrap=True)
    table.add_column("Tags", justify="left", style="red", no_wrap=False)
    table.add_column("Status", justify="center", style="green", no_wrap=False)
    table.add_column("Runtime", justify="center", style="white", no_wrap=False)
    table.add_column("Metric", justify="right", style="cyan", no_wrap=False)
    table.add_column("Commit sha", justify="right", style="white", no_wrap=True, width=10)
    table.add_column("Path", justify="right", style="white", no_wrap=True, width=10)

    for line in parsing_data.iterrows():       
        table.add_row(*line[1].tolist())

    console = Console()
    console.print(table)

    return parsing_data



def action_filter_by_tag(parsing_data):

    rprint("[bold red]Filtering with tags will modify the database locally. To retrieve all original experiments quit the prompt and launch again.[/bold red]")

    # Fetching tags to keep
    tags = input("Enter tags to keep separated by a comma: ")
    tags = [element.strip() for element in tags.split(',')]

    # Filtering parsing data with tags  wanted
    indexes_tag = parsing_data['Tags'].str.contains(tags[0])
    for tag in tags[1:]:
        indexes_tag = indexes_tag | parsing_data['Tags'].str.contains(tag)
    
    parsing_data = parsing_data.loc[indexes_tag]

    return parsing_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default='./results', help="Results directory")
    args = parser.parse_args()

    parsing_data = parse_all_experiments(args.results_dir)

    choice = main_menu()
    while choice != 4:
        if choice == 0:
            parsing_data = action_show_all_experiments(parsing_data)
        elif choice == 1:
            parsing_data = action_filter_by_tag(parsing_data)
            action_show_all_experiments(parsing_data)
        elif choice == 2:
            parsing_data = print("TODO")
        elif choice == 3:
            action_show_one_experiment(parsing_data)
        
        choice = main_menu()
    
