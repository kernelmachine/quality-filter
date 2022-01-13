from tqdm.auto import tqdm
import numpy as np


def score_text(df, clf, clf_vectorizer, field='text'):
    ## score text using quality filter
    df['filter_output']  = clf.predict_proba(clf_vectorizer.transform(tqdm(df[field]))).tolist()
    df['prob_low_quality'] = df.filter_output.apply(lambda x: x[0])
    df['prob_high_quality'] = df.filter_output.apply(lambda x: x[1])
    df = df.drop(['filter_output'], axis=1)
    df['GPT3_included'] = df.prob_high_quality.apply(lambda x: np.random.pareto(9) > (1 - x))

    return df

def get_counts(df, field='text'):
    # count number of whitespace tokens
    df['num_tokens'] = df[field].progress_apply(lambda x: len(x.split()))
    return df

def score(x):
    # score a single document
    return clf.predict_proba(clf_vectorizer.transform([x]))