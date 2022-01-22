
import argparse
from tqdm.auto import tqdm
import pandas as pd


def count_mentions(word):
    df = pd.read_json("/projects/arkdata/sg01/openwebtext/openwebtext.sample.jsonl.gz", lines=True, chunksize=10000)
    num_mentions = 0

    for chunk in tqdm(df):
        n_mention = chunk.text.apply(lambda x: counter(word, x)).sum()
        num_mentions += n_mention


    return num_mentions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--word")
    args = parser.parse_args()
    print(count_mentions(args.word))