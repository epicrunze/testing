import matplotlib.pyplot as plt
from numpy.ma.core import get_data
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from functools import partial

def main(MCL_file, target_cols, target_optims):
    results = []
    for root, dirs, files in os.walk("ResNet18"):
        for file in files:
            results.append(os.path.join(root, file))
    
    results_dict = defaultdict(list)

    for file in results:
        results_dict[os.path.basename(os.path.dirname(file)).replace("_StepDecay", "")].append(file)
    
    results_dict = {key: value for key, value in results_dict.items() if key in target_optims}

    results_dict = {key: average_all_dicts(
                            [average_valid_keys(
                                dict_to_array(
                                    get_data_dict(
                                        path_to_file=path2file, 
                                        target_cols=target_cols)
                                    )
                                ) 
                            for path2file in file_path_list]
                            ) 
                    for key, file_path_list in results_dict.items()}

    # adding MCL to dict
    results_dict["MCL"] = average_valid_keys(
                                dict_to_array(
                                    get_data_dict(MCL_file,
                                              target_cols)))

    results_dict = reorganize_dicts(results_dict)

    print(results_dict.keys())

    for metric, layers_dict in results_dict.items():
        for layer, optim_dict in layers_dict.items():
            optim_legend = []
            for optim, data in optim_dict.items():
                plt.plot(np.arange(data.shape[0]) + 1, data)

                # adding legend
                optim_legend.append(optim)

            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.legend(optim_legend)
            plt.title(f"{metric} for Layer {layer}")
            plt.savefig(metric + str(layer) + ".png")
            plt.clf()
    
    
def reorganize_dicts(optim_to_value):
    out_dict = defaultdict(partial(defaultdict, defaultdict))
    for optim, value in optim_to_value.items():
        for metric, array in value.items():
            for i, subarray in enumerate(np.split(array, array.shape[1], axis=1)):
                out_dict[metric][i+1][optim] = subarray.flatten()
    return out_dict

def get_data_dict(path_to_file, target_cols):
    data = pd.read_excel(path_to_file)
    results = defaultdict(list)
    for col in data.columns:
        if col.split("_epoch_")[0] in target_cols:
            results[col.split("_epoch_")[0]].append(data[col].tolist())
    
    return results

def dict_to_array(results_dict):
    """ returns a {value_of_interest: np.array(shape=(num_epochs by layers))} """
    return {key: np.array(value) for key, value in results_dict.items()}

def average_valid_keys(array_dict):
    if "in_S" in array_dict.keys() and "out_S" in array_dict.keys():
        array_dict["KG"] = np.mean(
            np.stack(
                (array_dict.pop("in_S"), array_dict.pop("out_S")), 
                axis=0), 
            axis=0)
    if "in_condition" in array_dict.keys() and "out_condition" in array_dict.keys():
        array_dict["condition_num"] = np.mean(
            np.stack(
                (array_dict.pop("in_condition"), array_dict.pop("out_condition")), 
                axis=0), 
            axis=0)
    return array_dict

def average_all_dicts(list_of_dicts):
    out_dict = defaultdict(list)
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            out_dict[key].append(value)
    
    for key, value in out_dict.items():
        out_dict[key] = np.mean(np.stack(value, axis=0), axis=0)
    
    return out_dict
    

if __name__ == "__main__":
    target_cols = ["in_S", "out_S", "in_condition", "out_condition"]
    target_optims = ["SGD", "MCL", "AdamP", 'Adas']
    target_file = "2021-07-09-08-10-44_resnet18_ADP-Release1_SGD_StepLR_15_0.5_LR=0.1_eta=0.01.xlsx"
    # results = get_data_dict(path2file, target_cols)
    # results = dict_to_array(results)
    # print(results.keys())
    # print([value.shape for value in results.values()])
    main(target_file, target_cols, target_optims)