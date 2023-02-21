'''
File: execute_experiment.py
Created Date: Mon Jan 09 2023
Author: Ammar Mian
-----
Last Modified: Thu Feb 02 2023
Modified By: Ammar Mian
-----
Copyright (c) 2023 Université Savoie Mont-Blanc
-----
File to actually execute an experiment. Prefer the use of launch_experiment to manage efficiently experiments.
'''

import argparse
import yaml, json
import logging
import os, sys
from importlib import import_module
from pathlib import Path
from datetime import datetime

from joblib import Parallel, delayed

from MIRAG.h5file_read import get_matrices, examine_h5
from MIRAG.utils import TqdmToLogger, StreamToLogger
from MIRAG.affichage import scale_0_1
from MIRAG.dictionary.dico_func import ricky, filtre2D_B

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

from itertools import product
import pickle


def inversion_one_algorithm(image, dictionary, estimation_dict, logger):
    logger.info(f"Doing method {estimation_dict['name']}")
    # Constructing the estimator object from the hyperparameters fetched from
    # the configuration file
    estimator = estimation_dict['estimator'](dictionary, **estimation_dict['hyperparameters'])
    estimator.fit(image)
    return np.real(estimator.get_estimate())


if __name__ == "__main__":
    
    # Managing execution
    # -------------------------------------------------------------------------------------------------------
    
    # Adding arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_file",  help="YAML experiment configuration file")
    parser.add_argument("directory_path", help="Results directory")
    args = parser.parse_args()

    # Parsing experiment configuration file
    with open(args.experiment_file,'rb')  as f:
        experiment = yaml.load(f, Loader=yaml.FullLoader)
    
    # Logging stuff
    logging.basicConfig(filename=os.path.join(args.directory_path, "experiment_execution.log"), 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        filemode='w') 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Stdout and stderr to logger as well
    sys.stdout = StreamToLogger(logger,logging.INFO)
    sys.stderr = StreamToLogger(logger,logging.ERROR)


    # Redirecting tqdm to logger
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)

    # Putting start time of experiment. Necessary in runner job case
    # Cause the job can start a long time after the submit
    metadata_file = os.path.join(args.directory_path, "running_metadata.yml")
    with open(metadata_file,'r') as f:
        running_metadata = yaml.load(f, Loader=yaml.FullLoader)
    running_metadata['start_date'] = datetime.now().strftime("%m-%d-%Y:%H-%M-%S")
    with open(metadata_file,'w') as f:
        yaml.dump(running_metadata, f)

    # Reading data image  from file
    # -------------------------------------------------------------------------------------------------------
    path_data = Path(experiment['data']['path'])
    logger.info(f'Reading Image data from {path_data}')
    if path_data.suffix == '.h5':
        raw, procs, masks = get_matrices(path_data)
        info = examine_h5(path_data)
        physical_parameters = json.loads(info['Raw_Data : Raw_Data_Array metadata'][0][1])

        if experiment['data']['type'] == "raw":
            image =  raw['Raw_Data_Array']
        else:
            image = procs['Processed_Image_001']

        mask = np.zeros_like(image)
        for mask_layer_name in masks.keys():
            mask = np.logical_or(mask, masks[mask_layer_name])

    elif path_data.suffix == '.pkl':
        with open(path_data, "rb") as f:
            image = pickle.load(f)
        # No mask in this case
        mask = None

        physical_parameters = {}

    N_t, N_x = image.shape

    
    # Constructing dictionary
    # -------------------------------------------------------------------------------------------------------
    if "only_dictionary" in experiment:
        only_dictionary = experiment["only_dictionary"]
    else:
        only_dictionary = False

    if experiment["dictionary"]["type"] == "physical":
        logger.info("Constructing atoms dictionary")
        duration_signal = physical_parameters["Length of trace (ns) : "] + physical_parameters['Startposition (ns) : '] # in ns
        length_signal = 0.0101 * N_x # in m
        dictionary_parameters = experiment["dictionary"]["parameters"]
        R_values = dictionary_parameters["R_values"]
        epsilon_values =  dictionary_parameters["epsilon_values"]
        number_atoms = len(R_values) * len(epsilon_values)

        if experiment["data"]["debug"]:
            image = image[:, 2000:3200]
            if mask is not None:
                mask = mask[:, 2000:3200]
            # duration_signal = duration_signal*350/N_t
            length_signal = length_signal*1200/N_x
            N_t, N_x = image.shape


        dictionary = np.zeros((N_t, N_x, number_atoms))
        for i, parameters in enumerate(tqdm(product(epsilon_values, R_values), total=number_atoms, file=tqdm_out)):
            epsilon, R = parameters
            atom = ricky(
                epsilon, R, duration_signal*1e-9, length_signal, N_x, N_t,
                physical_parameters['Antenna frequency (MHz) : ']*1e6
            )

            # Putting a hanning window to accoutn for loss in depth
            A = filtre2D_B(
                N_x, N_t, int(N_x/2), int(N_t/4), coef=dictionary_parameters['hanning_coef']
            )
            atom = A*atom
            dictionary[:,:,i] = atom/np.linalg.norm(atom, "fro")

    elif experiment["dictionary"]["type"] == "pre-computed":
        logger.info(f'Reading dictionary from {experiment["dictionary"]["path"]}')
        with open(experiment["dictionary"]["path"], "rb") as f:
            dictionary = pickle.load(f)

    else:
        logger.error(f'Dictionary type {experiment["dictionary"]["type"]} unknown. Exitting.')
        exit(0) 


    # Launching inversion methods in parallel 
    # -------------------------------------------------------------------------------------------------------

    if not only_dictionary:
        # Fetching inversion methods objects
        list_methods  = experiment['methodologies'].keys()
        logger.info(f"Doing inversion for all algorithms: {list(list_methods)}")

        for method in list_methods:
            # Fetching algorithm class object from the code file in YAML
            model = experiment['methodologies'][method]['model_file']
            module = import_module(model, package="experiments")
            experiment['methodologies'][method]['estimator'] = module.estimator
            
        # Doing the inversion
        n_jobs = -1
        logger.info(f"Number of available cpu threads: {os.cpu_count()}")
        logger.info(f"Executing with n_jobs={n_jobs}")
        results = Parallel(n_jobs=n_jobs)(
        delayed(inversion_one_algorithm)(
            image, dictionary,
            experiment['methodologies'][method], logger)
            for method in list_methods
        )


        # Fetching and formatting all results in a dict to pickle it as an artifact
        if mask is not None:
            results_dictionary = {
                'only_dictionary': only_dictionary,
                'image': image,
                'physical parameters': physical_parameters,
                'mask': mask,
                'dictionary': dictionary,
                'denoised_images': {},
                'ROC number of points': experiment['ROC']['number_points'],
                'ROC scale': experiment['ROC']['scale'],
                'ROC': {},
                'AUC': {}
            }
        else:
            results_dictionary = {
                'only_dictionary': only_dictionary,
                'image': image,
                'physical parameters': physical_parameters,
                'mask': mask,
                'dictionary': dictionary,
                'denoised_images': {},
            }

        
        # Computing ROC curve with associated mask if mask exists
        for image_denoised, method in zip(results, list_methods):
            method_name = experiment['methodologies'][method]['name']
            results_dictionary['denoised_images'][method_name] = image_denoised

            if mask is not None:
                a = np.array(image_denoised, dtype=np.float64)**2
                b = mask
                b = b.ravel()
                a = a.ravel()
                auc_score = roc_auc_score(b, a)
                fpr, tpr, thresholds = roc_curve(b, a)

                results_dictionary['AUC'][method_name] = auc_score
                results_dictionary['ROC'][method_name] = {'FPR': fpr, 'TPR':tpr}
    else:
        results_dictionary = {
                'only_dictionary': only_dictionary,
                'image': image,
                'physical parameters': physical_parameters,
                'mask': mask,
                'dictionary': dictionary,
        }

    # Saving artifact
    # -------------------------------------------------------------------------------------------------------
    with open(os.path.join(args.directory_path, "artifact.pkl"), 'wb') as f:
        pickle.dump(results_dictionary, f)


    # Managin end of execution through metadata file
    # -------------------------------------------------------------------------------------------------------
    logger.info("Handling end of job tasks.")
    logger.info(f"Saving run metadata in {metadata_file}")
    date_end = datetime.now()
    running_metadata['end_date'] = date_end.strftime("%m-%d-%Y:%H-%M-%S")

    
    if mask is not None and not only_dictionary:
        running_metadata['metric'] =  str(results_dictionary['AUC'])
    else:
        running_metadata['metric'] = "undefined"

    with open(metadata_file,'w') as f:
        yaml.dump(running_metadata, f)

    logger.info(f"Metric: {running_metadata['metric']}")
    logger.info("Experiment endded normally.")
