import pandas as pd
from data.score import get_counts, score_text
from tqdm.auto import tqdm
from lr.eval import load_model
from data.constants import DATA_DIR
clf, clf_vectorizer = load_model(DATA_DIR / "new_model/")


def load_and_score(path="articles-high-reliability-clean.jsonl"):
    news = pd.read_json(path, lines=True).drop_duplicates(subset=['text'])
    news = score_text(news, clf, clf_vectorizer)
    news = get_counts(news)
    return news


def load_school_news_data(path_to_articles='../logistic_regression/school_newspapers/high_school_articles_with_scores.jsonl', 
              path_to_metadata='../logistic_regression/school_newspapers/school_full_info_with_votes.jsonlist'):
    # read data
    school_newspapers = pd.read_json(path_to_articles, lines=True)
    # compute token counts
    school_newspapers = get_counts(school_newspapers)
    # read metadata
    papers_metadata = pd.read_json(path_to_metadata, lines=True)
    df = school_newspapers.merge(papers_metadata, on='domain')

    # merge some id columns
    df['school:county:state'] = df['school_name'].str.lower() + ":" + df['county'].str.lower() + ":" + df['state'].str.lower()
    df['zipcode:county:state'] = df['zipcode'].astype(str) + ":" + df['county'].str.lower() + ":" + df['state'].str.lower()
    df['county:state'] = df['county'].str.lower() + ":" + df['state'].str.lower()

    # filter down to only the most popular schools (> 100 documents)
    school_counts = df['school:county:state'].value_counts()
    populated_schools = school_counts.loc[school_counts > 100].index
    df = df.loc[df['school:county:state'].isin(populated_schools)]

    # filter down to data from only the past decade
    df['date'] = df.date.str.split(',').apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    df = df.loc[df.date.isin(['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019'])]
    return df


def get_high_schools(df):
    # get data from high schools only
    high_schools = df.loc[df.school_type_x == 'high'].copy()
    tqdm.pandas()
    # remove any content in the following categories (usually degenerate data)
    def check_cat(x, categories = None):
        bad_categories = ['video','multimedia','photos', 'media', 'videos','photo-of-the-day','galleries','photography','multimedia/video','photo-galleries']
        return not any([y in x for y in bad_categories])
    high_schools = high_schools.loc[high_schools.category.progress_apply(check_cat)]
    return high_schools


def compute_stats(df):
    # compute basic statistics of the data across the U.S. geography
    stats_ = {'high': {}}
    for school_type in tqdm(['high']):
        for stat_type in tqdm(['text', 'school:county:state', 'zipcode:county:state', 'county:state', 'state'], leave=False):
            k = df[stat_type].dropna()
            if k.dtype != int:
                k = k.loc[k.apply(len) > 1]
            stats_[school_type][stat_type] = k.unique().shape[0]
    return pd.DataFrame(stats_).T