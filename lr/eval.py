import argparse
import json
import logging
import os
import sys
import pathlib
import random
import shutil
import time
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer, HashingVectorizer,
                                             TfidfVectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from lr.hyperparameters import SEARCH_SPACE, RandomSearch, HyperparameterSearch
from shutil import rmtree


# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_model(serialization_dir):
    with open(os.path.join(serialization_dir, "best_hyperparameters.json"), 'r') as f:
        hyperparameters = json.load(f)
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
    elif weight == 'hash':
        vect = HashingVectorizer(stop_words=stop_words,lowercase=True,ngram_range=ngram_range)
    else:
        vect = CountVectorizer(binary=binary,
                               stop_words=stop_words,
                               lowercase=True,
                               ngram_range=ngram_range)
    if weight != "hash":
        with open(os.path.join(serialization_dir, "vocab.json"), 'r') as f:
            vocab = json.load(f)
        vect.vocabulary_ = vocab
    hyperparameters['C'] = float(hyperparameters['C'])
    hyperparameters['tol'] = float(hyperparameters['tol'])
    classifier = LogisticRegression(**hyperparameters)
    if os.path.exists(os.path.join(serialization_dir, "archive", "idf.npy")):
        vect.idf_ = np.load(os.path.join(serialization_dir,  "archive", "idf.npy"))
    classifier.coef_ = np.load(os.path.join(serialization_dir,  "archive", "coef.npy"))
    classifier.intercept_ = np.load(os.path.join(serialization_dir,  "archive", "intercept.npy"))
    classifier.classes_ = np.load(os.path.join(serialization_dir,  "archive", "classes.npy"))
    return classifier, vect


def eval_lr(test,
            classifier,
            vect,
            text_field='text'):
    start = time.time()
    X_test = vect.transform(tqdm(test[text_field], desc="fitting and transforming data"))
    end = time.time()
    preds = classifier.predict(X_test)
    scores = classifier.predict_proba(X_test)
    return scores

def batchify(data, batch_size: int):
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str)
    parser.add_argument('--eval_name', type=str, default=None)
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--output', '-o', type=str)
    
    
    
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print(f"model {args.model} does not exist. Aborting! ")
    else:
        clf, vect = load_model(args.model)



    if args.eval_file == '-':
        while True:
            for line in iter(sys.stdin.readline, ''):
                try:
                    line = json.loads(line)
                    out = clf.predict_proba(vect.transform([line['text']]))
                    print(f"{line['url']},{out[0][1]}")
                except:
                    print("n/a,0.0")
              
    else: 
    	print(f"reading evaluation data at {args.eval_file}...")
    	test = pd.read_json(args.eval_file, lines=True)

    	scores = eval_lr(test, clf, vect, "raw_content")
    	if args.output:
        	out = pd.DataFrame({'uel': test['url'], 'score': scores.tolist()})
        	out.to_json(args.output, lines=True, orient='records')

        	#print("================")
        	#print(f"F1: {f1}")
        	#print(f"accuracy: {acc}")
