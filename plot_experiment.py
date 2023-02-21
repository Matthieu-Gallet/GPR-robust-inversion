'''
File: plot_experiment.py
Created Date: Thu Jan 01 1970
Author: Ammar Mian
-----
Last Modified: Tue Jan 17 2023
Modified By: Ammar Mian
-----
Copyright (c) 1970-2023 Université Savoie Mont-Blanc
-----
File detailling how to plot an experiment from the data files stored
in experiment directory
'''

import os
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt
MIRAG.affichage import scale_0_1
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
import json

if __name__ == "__main__":
    # Adding arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_results_directory",  help="Experiment results directory path")
    parser.add_argument("--with_latex", action="store_true", default=False)
    args = parser.parse_args()

    if args.with_latex:
        from matplotlib import rc
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        rc('text', usetex=True)
        __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    # Open artifact for plotting
    print("Reading data from: ", os.path.join(args.experiment_results_directory, "artifact.pkl"))
    with open(os.path.join(args.experiment_results_directory, "artifact.pkl"), 'rb') as f:
        results_dictionary = pickle.load(f)

    if 'only_dictionary' in results_dictionary:
        only_dictionary = results_dictionary['only_dictionary']
    else:
        only_dictionary = False

    if 'physical parameters' in results_dictionary.keys():
        print("Physical parameters of the image:")
        pretty = json. dumps (results_dictionary['physical parameters'], indent=4)
        print (pretty)

    print("Plotting Images")
    plt.figure(figsize=(8,4))
    plt.imshow(scale_0_1(results_dictionary['image']), cmap="gray", aspect="auto")
    plt.colorbar()
    plt.title("Image")
    plt.tight_layout()

    if not only_dictionary:
        # Plotting denoised images
        print("Plotting Denoised images")
        for method in results_dictionary['denoised_images'].keys():
            plt.figure(figsize=(8,4))
            plt.imshow(scale_0_1(results_dictionary['denoised_images'][method]), cmap="gray", aspect="auto")
            plt.colorbar()
            plt.title(f"Results of inversion: {method}")
            plt.tight_layout()
        
        if results_dictionary['mask'] is not None:

            # Plot mask
            print('Plottin mask of hyperbolas')
            plt.figure(figsize=(8,4))
            plt.imshow(results_dictionary['mask'], cmap="gray", aspect="auto")
            plt.colorbar()
            plt.title("Mask of hyperbolas")
            plt.tight_layout()

            # Plotting ROC curves
            print("Plotting ROC curves")
            plt.figure(figsize=(7,4))
            for method in results_dictionary['ROC'].keys():
                fpr = results_dictionary['ROC'][method]['FPR']
                tpr = results_dictionary['ROC'][method]['TPR']

                if results_dictionary['ROC scale'] == 'linear':
                    indexes = np.unique(np.linspace(0, len(fpr)-1, results_dictionary['ROC number of points'], dtype=int))
                else:
                    indexes = np.unique(np.logspace(0, np.log10(len(fpr)-1), results_dictionary['ROC number of points'], dtype=int))
                
                fpr, tpr  = fpr[indexes], tpr[indexes]

                if results_dictionary['ROC scale'] == "linear":
                    plt.plot(fpr, tpr, label=f'{method} - ROC curve (area = %0.2f)' % results_dictionary['AUC'][method])
                else:
                    plt.semilogx(fpr, tpr, label=f'{method} - ROC curve (area = %0.2f)' % results_dictionary['AUC'][method])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend()
                plt.tight_layout()

    # Showing dictionary with a slider:
    print("Plotting hyperbolas dictionary")
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 2, height_ratios=[8,1], width_ratios=[6,0.5])
    ax_image, ax_slider = plt.subplot(gs[0, 0]), plt.subplot(gs[1,:])
    ax_colorbar = plt.subplot(gs[0, 1])
    im = ax_image.imshow(results_dictionary['dictionary'][:,:,0], aspect="auto", cmap="gray")
    ax_image.set_title("Atom 0")

    fig.colorbar(im, cax=ax_colorbar, orientation="vertical")
    slider = Slider(ax_slider, 'Slide->', 0, results_dictionary['dictionary'].shape[-1]-1, valinit=0)
    def update(val):
        ax_image.imshow(results_dictionary['dictionary'][:,:,int(val)], cmap='gray', aspect="auto")
        ax_image.set_title(f"Atom {int(val)}")
        fig.colorbar(im, cax=ax_colorbar, orientation="vertical")
        fig.canvas.draw_idle()
    slider.on_changed(update)

    plt.show()
