#!/usr/bin/env python
from genheas import generate
from genheas import train


def main():
    _, best_policy_file = train.main()
    generate.main(best_policy_file)


if __name__ == "__main__":
    main()
