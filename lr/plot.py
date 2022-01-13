import argparse
import json
import logging
import os
import pathlib
import random
import shutil
import time
from typing import Any, Dict, List, Union
import seaborn as sns
import sys
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd



# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_model(hyperparameters):

    if hyperparameters.pop('stopwords') == 1:
        stop_words = 'english'
    else:
        stop_words = None
    weight = hyperparameters.pop('weight')
    if weight == 'binary':
        binary = True
    else:
        binary = False
    ngram_range = hyperparameters.pop('ngram_range')
    ngram_range = sorted([int(x) for x in ngram_range.split()])
    if weight == 'tf-idf':
        vect = TfidfVectorizer(stop_words=stop_words,
                               lowercase=True,
                               ngram_range=ngram_range)
    else:
        vect = CountVectorizer(binary=binary,
                               stop_words=stop_words,
                               lowercase=True,
                               ngram_range=ngram_range)
    hyperparameters['C'] = float(hyperparameters['C'])
    hyperparameters['tol'] = float(hyperparameters['tol'])
    classifier = LogisticRegression(**hyperparameters)
    return classifier, vect


def eval_lr(test,
            classifier,
            vect):
    start = time.time()
    X_test = vect.fit_transform(tqdm(test.text, desc="fitting and transforming data"))
    end = time.time()
    preds = classifier.predict(X_test)
    return f1_score(test.label, preds, average='macro'), classifier.score(X_test, test.label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', '-m', type=str)
    parser.add_argument('--performance_metric', '-p', type=str)
    parser.add_argument('--hyperparameter', '-x', type=str)
    parser.add_argument('--logx', action='store_true')
    parser.add_argument('--boxplot', action='store_true')

    
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"Results file {args.results_file} does not exist. Aborting! ")
        sys.exit(1)
    else:
        df = pd.read_json(args.results_file, lines=True)
    if args.boxplot:
        ax = sns.boxplot(df[args.hyperparameter], df[args.performance_metric])
    else:
        ax = sns.scatterplot(df[args.hyperparameter], df[args.performance_metric])
    if args.logx:
        ax.set_xscale("log")
    plt.show()