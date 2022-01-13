import argparse
import json
import logging
import os
import pathlib
from typing import Any, Dict, List, Union
import sys
import pandas as pd

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', nargs="+", type=str)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    dfs = []
    for experiment in args.experiments:
        if not os.path.isdir(experiment):
            print(f"experiment {experiment} does not exist. Aborting! ")
            sys.exit(1)
        else:
            dfs.append(pd.read_json(os.path.join(experiment, "results.jsonl"), lines=True))
    master = pd.concat(dfs, 0)
    master.to_json(args.output, lines=True, orient='records')