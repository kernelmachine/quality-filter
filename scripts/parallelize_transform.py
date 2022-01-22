from lr.eval import load_model
clf, clf_vectorizer = load_model("../logistic_regression/new_model/")
import numpy as np
import pandas as pd
from multiprocessing import cpu_count, Pool
from tqdm.auto import tqdm
import argparse

cores = 10 #Number of CPU cores on your system
partitions = cores #Define as many partitions as you want

def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    result = []
    for x in tqdm(pool.imap(func, data_split), total=len(data_split), leave=False):
        result.append(x)
    data = np.concatenate(result).tolist()
    pool.close()
    pool.join()
    return data
def score_text(df, field='text'):
    df['filter_output']  = parallelize(df.text, score)
    df['prob_low_quality'] = df.filter_output.apply(lambda x: x[0])
    df['prob_high_quality'] = df.filter_output.apply(lambda x: x[1])
    df = df.drop(['filter_output'], axis=1)
    return df


def score(x):
    return clf.predict_proba(clf_vectorizer.transform(x))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')
    parser.add_argument('--chunksize', type=int)
    
    args = parser.parse_args()
    print(f"reading datai from {args.input_path}...")
    chunks = pd.read_json(args.input_path, lines=True, chunksize=args.chunksize)
    for i, chunk in tqdm(enumerate(chunks)):
        chunk = score_text(chunk)
        chunk.to_json(args.output_path + "/" + str(i), lines=True, orient='records')
