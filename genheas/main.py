#!/usr/bin/env python
import argparse
import sys

from genheas import generate
from genheas import train


parser = argparse.ArgumentParser(description="Molecular Modelling Control Software")
parser.add_argument(
    "config_options",
    metavar="OPTIONS",
    nargs="+",
    help="configuration options, the path to  dir whit configuration file",
)


def main(root_dir):
    _, best_policy_file = train.main(root_dir)
    generate.main(root_dir, best_policy_file)


if __name__ == "__main__":
    # args = parser.parse_args()
    args = parser.parse_args(sys.argv[1:])

    workdir = args.config_options[0]
    main(workdir)
