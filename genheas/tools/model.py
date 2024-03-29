#!/usr/bin/env python
"""
Created on Fri Oct 16 23:27:41 2020

@author: Conrard TETSASSI
"""
import datetime
import glob
import json
import multiprocessing as mp
import os
import pathlib
import pickle
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml

from genheas.tools.feedforward import Feedforward
from genheas.tools.gencrystal import AlloysGen
from genheas.tools.gencrystal import coordination_numbers
from genheas.tools.neural_evolution import NeuralEvolution
from genheas.tools.properties import Property
from genheas.tools.properties import atomic_properties
from genheas.tools.properties import atomic_properties_categories
from genheas.utilities.log import logger
from pymatgen.io.ase import AseAtomsAdaptor


# import torch.nn.init as init

# from math import prod


# from genheas.optimize import output
# Scheduler import
# from torch.optim.lr_scheduler import StepLR

# import EarlyStopping
# from pytorchtools import EarlyStopping

# Set seed
# torch.manual_seed(0)

best_mae_error = 1e10
best_fitness = 0.0


def save_checkpoint(state, is_best, filename, best_name):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name)


sns.set()

opt_parameters_list = [
    "nb_atom",
    "device",
    "rate",
    "alpha",
    "nb_generation",
    "nb_policies",
    "nb_network_per_policy",
    "opt_time",
    "input_size",
    "output_size",
    "step_time",
    "cell_size",
    "cell_param",
    "cubik",
]


def plot_optimization(data_filepath, skiprows=1):
    """
    data_filepath = "training_data/{}a-{}p-{}n-{}m.csv".format(alpha,nb_policies,nb_network_per_policy)

    By default we skip the first row, which contains the headers
    By skipping 2 rows, you can disregard the first data-point (0,0) to get
    a closer look
    :param data_filepath:
    :param skiprows:
    :return:
    """
    plt.close()
    for filename in glob.glob(data_filepath + "/*.csv"):
        x, y = np.loadtxt(filename, delimiter=",", unpack=True, usecols=(0, 1), skiprows=skiprows)
        plt.plot(x, y, label=os.path.basename(filename).split(".csv")[0], linewidth=3)

    plt.xlabel("Generation", fontsize=16)
    plt.ylabel("MAE", fontsize=16)
    plt.title("Optimization Curve", fontsize=20)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)
    plt.legend(fontsize=15, loc="upper right")
    plt.savefig(f"{data_filepath}/optimization_curve.png", dpi=600, transparent=True, bbox_inches="tight")
    # plt.show()


def training(alloy_gen, networks, structure, element_pool, device="cpu"):
    """
    :param  networks:
    :param alloy_gen:
    :param structure:
    :param element_pool:
    :param device:
    :return:
    """

    config = alloy_gen.generate_configuration(structure, element_pool, networks, max_diff_element=None, device=device)

    return config


def multiprocessing_training(workers, alloy_gen, list_networks, list_structures, element_pool):
    # logger.info('multiprocessing training')
    pool = mp.Pool(workers)
    configs = pool.starmap(
        training,
        [(alloy_gen, net, struc, element_pool) for net, struc, in zip(list_networks, list_structures)],
    )
    pool.close()
    return configs


def serial_training(NEs, list_networks, list_structures, element_pool):
    # logger.info('serial training')
    configs = [training(NEs, net, struc, element_pool) for net, struc, in zip(list_networks, list_structures)]
    return configs


def train_policy(
    crystal_structure,
    element_pool,
    concentrations,
    nb_generation,
    cell_size,
    cell_param,
    device="cpu",
    rate=0.25,
    alpha=0.1,
    nn_per_policy=1,
    nb_policies=8,
    nb_worker=1,
    best_fitness=0.5,
    patience=100,
    cubik=False,
):
    """
    :param nb_worker:
    :param cubik:
    :param patience: (int) nb generations to wait before early stop
    :param best_fitness: (float) minimum fitness to reach
    :param nb_generation: (int) number of generation to run
    :param nb_policies: (int) number of policies = input structures
    :param nn_per_policy: (int) number of network per policy
    :param alpha: (float) scaling factor for the weight mutation
    :param rate: (float) % of policy to keep at each generation
    :param device: cpu or gpu
    :param cell_param: (array) lattice constant of the input structure [3.61, 3.61, 3.61]
    :param cell_size: (array) size of the input structure  [4,4,4]
    :param concentrations: (dict) concentration of each element {'Ag': 0.5, 'Pd': 0.5}
    :param element_pool: list of element ['Ag', 'Pd']
    :param crystal_structure: (str) 'fcc' 'bcc' 'hpc'
    :return: list with the best weight
    """
    # ==========================  Initialization  ============================
    global best_mae_error
    logger.info(f"Input parameters:: alpha [{alpha}]\t rate [{rate}] \t device [{device}]")
    # early_stop = False
    n_generations_stop = patience
    generations_no_improve = 0

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NEs = NeuralEvolution(element_pool, concentrations, crystal_structure, rate=rate, alpha=alpha, device=device)

    alloy_gen = AlloysGen(element_pool, concentrations, crystal_structure, radius=8.0)

    # combinations = alloy_gen.get_peers(element_pool)

    # nb_species = len(element_pool)
    workdir = pathlib.Path.cwd()
    output = os.path.join(workdir, "_".join(element_pool), "generations_" + str(nb_generation))

    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    nn_in_shell1 = coordination_numbers[crystal_structure][0]
    nn_in_shell2 = coordination_numbers[crystal_structure][1]
    n_neighbours = nn_in_shell1 + 1  # + nn_in_shell2

    input_size = n_neighbours * len(atomic_properties.values())

    output_size = len(element_pool)

    sorted_weights_list = None
    # iter = 0
    min_fitness = []
    nb_network_per_policy = nn_per_policy
    nb_policies = nb_policies

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # my_model = Feedforward(input_size, output_size)
    # my_model.to(device)
    since0 = time.time()
    nb_atom = len(
        alloy_gen.gen_crystal(
            crystal_structure,
            cell_size,
            max_diff_elem=None,
            lattice_param=None,
            name=None,
            cubik=cubik,
            radom=False,
        ),
    )

    max_diff_elements = alloy_gen.get_max_diff_elements(nb_atom)
    NEs.get_target_shell1()
    NEs.get_target_shell2()
    # if not nb_atom == sum(max_diff_elements.values()):
    #     logger.error(
    #         "the size : [{}] and the max_diff_elem : [{}] are not consistent".format(
    #             nb_atom,
    #             sum(max_diff_elements.values()),
    #         ),
    #     )
    #     raise Exception(
    #         "the size : [{}] and the max_diff_elem : [{}] are not consistent".format(
    #             nb_atom,
    #             sum(max_diff_elements.values()),
    #         ),
    #     )

    logger.info("Generating the Input structure")
    n_input = nb_policies

    logger.info(f"max_diff_elements: {max_diff_elements}")
    logger.info(f"Number of policies: {n_input}")
    logger.info(f"Number of ANN per policy: {nn_per_policy}")
    logger.info(f"Number of atoms in the  input structure: {nb_atom}")

    logger.info("Start Optimization")
    # with torch.set_grad_enabled(False):
    networks = None

    input_structures = [
        alloy_gen.gen_crystal(
            crystal_structure,
            cell_size,
            max_diff_elem=None,
            lattice_param=cell_param,
            name=element_pool[0],
            cubik=cubik,
            radom=False,
        )
        for _ in range(n_input)
    ]

    for generation in range(nb_generation):
        since = time.time()

        if networks is None:  # first iteration
            # --------------------------------------------------------------------
            configurations = []
            networks_list = []

            for _ in range(len(input_structures)):
                networks = [Feedforward(input_size, output_size) for i in range(nb_network_per_policy)]
                networks = [network.to(device) for network in networks]
                # network_weights = [network.l1.weight.data for network in networks]
                networks_list.append(networks)
                # network_weights_list.append(network_weights)

            assert 1 <= nb_worker <= mp.cpu_count(), "1 <= nb_worker <= max_cpu"

            if nb_worker == 1:
                configurations = serial_training(alloy_gen, networks_list, input_structures, element_pool)
            else:
                configurations = multiprocessing_training(
                    nb_worker,
                    alloy_gen,
                    networks_list,
                    input_structures,
                    element_pool,
                )

            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            configurations_fitness = NEs.get_population_fitness(configurations, max_diff_elements)

            # Rank the policies

            sorted_policies = NEs.sort_policies_by_fitness(
                networks_list,
                configurations_fitness,
            )  # list of list

        else:
            networks_list = NEs.update_policies(sorted_policies, rate=rate)

            # ---------------------- multi process-----------------------------
            if nb_worker == 1:
                configurations = serial_training(alloy_gen, networks_list, input_structures, element_pool)
            else:
                configurations = multiprocessing_training(
                    nb_worker,
                    alloy_gen,
                    networks_list,
                    input_structures,
                    element_pool,
                )

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            configurations_fitness = NEs.get_population_fitness(configurations, max_diff_elements)

            # Rank the policies

            sorted_policies = NEs.sort_policies_by_fitness(
                networks_list,
                configurations_fitness,
            )  # list of list

        # save the current training information

        step_time = time.time() - since
        mae_error = np.min(configurations_fitness)
        min_fitness.append([generation + 1, mae_error, step_time])

        logger.info(
            "generation: {:6d}/{} |MAE {:10.6f} *** time/step: {:.1f}s".format(
                generation + 1,
                nb_generation,
                mae_error,
                step_time,
            ),
        )

        is_best = mae_error < best_mae_error

        best_mae_error = min(mae_error, best_mae_error)
        check_file = f"{output}/checkpoint.pth.tar"
        best_file = f"{output}/best_model_{alpha}a-{nb_policies}p-{nb_network_per_policy}n.pth.tar"
        state = {
            "nb_network_per_policy": nb_network_per_policy,
            "generation": generation + 1,
            "best_mae_error": best_mae_error,
            "alpha": alpha,
            "device": device,
        }

        for j, network in enumerate(sorted_policies[0]):  # nb_network_per_policy
            name = f"network_{j}"
            state[name] = network.state_dict()
        save_checkpoint(state, is_best, check_file, best_file)

        if mae_error <= best_fitness:

            generations_no_improve = 0
            best_fitness = mae_error
        else:
            generations_no_improve += 1

        if generation > 5 and generations_no_improve == n_generations_stop:
            logger.warning("Early stopping!")

            break
        else:
            continue

    logger.info("Optimization completed")
    best_policy = sorted_policies[0]  # sorted_weights_list[0]

    # BestWeights_file = f"{output}/BestWeights_{alpha}a-{nb_policies}p-{nb_network_per_policy}n.pkl"
    # pickle.dump(best_policy, open(BestWeights_file, "wb"))

    # logger.info(f'Best policy saved in  "{BestWeights_file}"')
    logger.info(f'Best policy saved in  "{best_file}"')
    min_fitness = np.array(min_fitness)
    # mean_fitness = np.array(mean_fitness)
    opt_time = time.time() - since0
    logger.info(
        "Cumulative optimization time after  {} generations [h:m:s] ::  {}".format(
            generation + 1,
            str(datetime.timedelta(seconds=opt_time)),
        ),
    )
    np.savetxt(
        f"{output}/{alpha}a-{nb_policies}p-{nb_network_per_policy}n.csv",
        min_fitness,
        delimiter=",",
        header="Generation, min_fitness, time",
    )
    # np.savetxt('{}/mean_fitness.csv'.format(output), mean_fitness, delimiter=",",
    #            header='Generation, min_fitness, time')

    plot_optimization(output, skiprows=1)

    # nb_atoms = len(input_structures[0])
    opt_parameters = {}
    for param in opt_parameters_list:
        opt_parameters[param] = eval(param)

    with open(f"{output}/opt_parameters_{alpha}a-{nb_policies}p-{nb_network_per_policy}n.yml", "w") as yaml_file:
        yaml.dump(opt_parameters, yaml_file, default_flow_style=False)

    # return sorted_weights_list[0], BestWeights_file
    return best_policy, best_file
