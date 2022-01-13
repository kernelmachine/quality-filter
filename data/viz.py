import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd


def build_school_category_plot(high_schools, news, save=False, save_path=None):
    f, axes = plt.subplots(2,1, figsize=(4,6))

    samples = []
    categories = [ 'announcements', 'campus-life', 'clubs','sports', 'op-ed', 'politics']
    for category in categories:
        cat= high_schools.loc[high_schools.category.apply(lambda x: category in x)]
        samples.append(cat.sample(min(cat.shape[0], news.shape[0]//6)))
    high_school_sample = pd.concat(samples, 0)
    ax = sns.histplot(news.prob_high_quality, label='Newswire', stat='density', color='#ffb347', alpha=0.6, ax=axes[0])
    ax = sns.histplot(high_school_sample.prob_high_quality, label='High School', stat='density', color='#00bfff', alpha=0.6, ax=axes[0])
    ax.set_xlabel("P(high quality)")

    #ffb347
    #00bfff
    ax.legend()


    dfs = []
    for category in categories:
        dfs.append(pd.DataFrame({"Article Category": category, 
                                "P(high quality)": high_schools.loc[high_schools['category'].apply(lambda x: category in x)].prob_high_quality}))


    sns.boxplot(data=pd.concat(dfs, 0), y='Article Category', x='P(high quality)', ax=axes[1], linewidth=2)
    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def boxplot(z, topics, save=False, save_path=None):
    grouped = z.groupby('cluster')
    df2 = pd.DataFrame({col:vals['prob_high_quality'] for col,vals in grouped})
    meds = df2.median()
    meds.sort_values(ascending=False, inplace=True)
    df2 = df2[meds.index]
    cols = df2.columns 
    ax = sns.boxplot(data=df2[df2.columns[::-1]], orient='h' ,linewidth=2)
    labels = []
    for i in ax.get_yticklabels():
        labels.append(str(int(float(i.get_text()))) + ":" + " ".join(topics.loc[topics.cluster == int(float(i.get_text()))].top_words.values[0]))
    _ = plt.xticks(rotation=90)
    ax.set_yticklabels(labels)
    _ = ax.set_xlabel("cluster")
    _ = ax.set_xlabel("P(high quality)")
    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def build_correlation_plots(features, save=False, save_path=None):
    sns.set(context='paper', style='white')

    fig, axes= plt.subplots(2,2, figsize=(6,6))
    feature = 'median_home_value'
    ax = sns.regplot(np.log(features[feature] + 1e-5),
                    features['prob_high_quality'],
                    ax=axes[0][0],
                    marker='o',
                    scatter_kws={'s':1},
                    line_kws={"color": "#ff6961"})
    ax.set_title("r: {:.2f}, p ≈ 0".format(stats.pearsonr(features[feature], features['prob_high_quality'])[0]))
    ax.set_ylabel("P(high quality)")
    ax.set_xlabel("Median Home Value")
    ax.set_xticks([np.log(50000),  np.log(100000), np.log(250000), np.log(500000), np.log(1000000)])
    ax.set_xticklabels(["50K", "100K", "250K", "500K", "1M"])


    feature = 'num_degree_holders'
    ax = sns.regplot(features[feature] * 100,
                    features['prob_high_quality'],
                    ax=axes[0][1],
                    marker='o',
                    scatter_kws={'s':1},
                    line_kws={"color": "#ff6961"})
    ax.set_title("r: {:.2f}, p ≈ 0".format(stats.pearsonr(features[feature], features['prob_high_quality'])[0]))
    ax.set_ylabel("P(high quality)")
    ax.set_xlabel("% Adults ≥ Bachelor Degrees")


    feature = 'rep_share_2016'

    ax = sns.regplot(features[feature] * 100,
                    features['prob_high_quality'],
                    ax=axes[1][0],
                    marker='o',
                    scatter_kws={'s':1},
                    line_kws={"color": "#ff6961"})
    ax.set_title("r: {:.2f}, p ≈ 0".format(stats.pearsonr(features[feature], features['prob_high_quality'])[0]))
    ax.set_ylabel("P(high quality)")
    ax.set_xlabel("% 2016 GOP Vote")
    # ax.set_xticks([np.log(1), np.log(10), np.log(100), np.log(1000), np.log(10000)])
    # ax.set_xticklabels(["1","10", "100", "1K", "10K"])

    feature = 'POPPCT_RURAL'
    ax = sns.regplot(features[feature],
                    features['prob_high_quality'],
                    ax=axes[1][1],
                    marker='o',
                    scatter_kws={'s':1},
                    line_kws={"color": "#ff6961"})
    ax.set_title("r: {:.2f},  p ≈ 0".format(stats.pearsonr(features[feature], features['prob_high_quality'])[0]))
    # ax.set_xscale('log')
    ax.set_ylabel("P(high quality)")
    ax.set_xlabel("% Rural")
    # ax.set_xticks([np.log(0.1), np.log(1), np.log(10), np.log(50)])
    # ax.set_xticklabels(["0.1", "1", "10", "50"])

    # ax.set_xticks([np.log(1), np.log(10), np.log(100), np.log(1000), np.log(10000)])
    # ax.set_xticklabels(["1","10", "100", "1K", "10K"])

    plt.tight_layout()
    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()