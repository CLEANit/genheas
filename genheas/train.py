#!/usr/bin/env python
"""
@Time    : 04 Jan 9:47 a.m. 2021
@Author  : Conrard Tetsassi
@Email   : giresse.feugmo@gmail.com
@File    : train.py
@Project : genheas
@Software: PyCharm
"""
import argparse
import os
import sys

from pathlib import Path

import torch
import yaml

from genheas import generate
from genheas.tools.feedforward import Feedforward
from genheas.tools.gencrystal import coordination_numbers
from genheas.tools.model import train_policy
from genheas.tools.properties import atomic_properties
from genheas.utilities.log import logger


parser = argparse.ArgumentParser(description="Molecular Modelling Control Software")
parser.add_argument(
    "config_options",
    metavar="OPTIONS",
    nargs="+",
    help="configuration options, the path to  dir whit configuration file",
)


def main(root_dir):
    # ========================== Read Parameters  ============================
    input_file = os.path.join(root_dir, "parameters.yml")

    try:
        with open(input_file) as fr:
            parameters = yaml.safe_load(fr)
    except Exception as err:
        logger.error(f"{err}")
        raise Exception(f"{err}")

    generations = parameters["nb_generation"]
    crystal_structure = parameters["crystalstructure"]
    cell_parameters = parameters["cell_parameters"]
    elements_pool = parameters["elements_pool"]
    concentrations = parameters["concentrations"]

    rate = parameters["rate"]
    alpha = parameters["alpha"]
    device = parameters["device"]

    nn_per_policy = parameters["nn_per_policy"]
    nb_policies = parameters["nb_policies"]
    nb_worker = parameters["nb_worker"]
    patience = parameters["patience"]

    training_size = parameters["cell_size"]
    cubic_train = parameters["cubic_train"]

    # ==========================  Analyze Parameters  ============================

    logger.info("Input parameters have been retrieved")

    nn_in_shell1 = coordination_numbers[crystal_structure][0]
    nn_in_shell2 = coordination_numbers[crystal_structure][1]
    n_neighbours = nn_in_shell1 + 1  # + nn_in_shell2

    input_size = n_neighbours * len(atomic_properties)

    output_size = len(elements_pool)

    workdir = Path.cwd()
    # ==============================  Training ===============================
    output = os.path.join(workdir, "_".join(elements_pool), "generations_" + str(generations))

    best_policy, best_policy_file = train_policy(
        crystal_structure,
        elements_pool,
        concentrations,
        generations,
        training_size,
        cell_parameters,
        rate=rate,
        alpha=alpha,
        nn_per_policy=nn_per_policy,
        nb_policies=nb_policies,
        nb_worker=nb_worker,
        patience=patience,
        cubik=cubic_train,
    )

    # ============================  Generation  ==========================

    # networks = [Feedforward(input_size, output_size) for _ in range(len(best_policy))]
    # networks = [network.to(device) for network in networks]
    #
    # for j, w in enumerate(best_policy):
    #     networks[j].l1.weight = torch.nn.Parameter(w)
    networks = best_policy
    generate.generate_structure(
        networks,
        output,
        elements_pool,
        concentrations,
        crystal_structure,
        training_size,
        cell_parameters,
        constraints=False,
        cubik=cubic_train,
    )
    return best_policy_file


if __name__ == "__main__":
    # args = parser.parse_args()
    args = parser.parse_args(sys.argv[1:])

    workdir = args.config_options[0]
    _, _ = main(workdir)
