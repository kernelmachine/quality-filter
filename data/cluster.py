from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from data.viz import boxplot


sns.set(style='white', font_scale=1.4, context='paper')

def cluster_text(z, num_clusters=10, num_words=10, plot_boxplot=False, save=False, save_path=None):
    cv = TfidfVectorizer(
        max_features=10000,
        min_df=3,
        stop_words="english")
    vecs = cv.fit_transform(z.text.str.lower())
    svd = TruncatedSVD(100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    vecs = lsa.fit_transform(vecs)
    km = KMeans(n_clusters=num_clusters)
    z['cluster'] = (pd.Series(km.fit_predict(vecs)))
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

    terms = cv.get_feature_names_out()
    clusters = []
    for i in range(num_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :num_words]]
        clusters.append({"cluster": i, 
                         "top_words": top_words,
                         "prob_high_quality": z.loc[z.cluster == i].prob_high_quality.mean()})

    clusters = pd.DataFrame(clusters)
    z['cluster'] = z.cluster.astype('category')
    if plot_boxplot:
        boxplot(z, clusters, save=save, save_path=save_path)
    return z

