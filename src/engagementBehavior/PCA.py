import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from .output import outputVariance, outputScatterPlot


def applyPCA(labeling_df, combined_rankings):
    # Pearson Correlation across Features
    labeling_df['mean_suspicion_rank'] = combined_rankings['suspicion rank - Mean Rank']
    labeling_df.drop(columns=['suspicion_rank(views_subs)', 'suspicion_rank(views_videos)',
        'suspicion_rank(views_comments)', 'suspicion_rank(subs_videos)',
        'suspicion_rank(subs_comments)', 'suspicion_rank(videos_comments)'], inplace=True)
    # Dimensionality Reduction
    feature_df = labeling_df.fillna(0)
    feature_df.rename(columns={'mean_suspicion_rank': "mean_suspicion_score"}, inplace=True)
    suspicion_df = feature_df['mean_suspicion_score']
    feature_df.drop(columns=['channel_id', 'start_date', 'end_date', 'duration(in days)', 'mean_suspicion_score'], axis=1, inplace=True)
    # Principal Component Analysis
    pca_12 = PCA(n_components=12, random_state=365)
    pca_12.fit(feature_df)
    _ = pca_12.transform(feature_df)
    outputVariance(pca_12)
    # Apply PCA using the first 2 Principal components
    pca_2 = PCA(n_components=2, random_state=365)
    pca_2.fit(feature_df)
    feature_df_pca_2 = pca_2.transform(feature_df)
    combined_df = getCombinedPCA(labeling_df, feature_df_pca_2)
    combined_df.to_csv('feature_pca.csv')
    outputScatterPlot(feature_df_pca_2, pca_12, suspicion_df)
    densityBasedClustering(feature_df_pca_2)


def getCombinedPCA(labeling_df, feature_df_pca_2):
    """Mapping PCA components to Original data. Create DataFrame with PCA and original Data."""
    combined_df = labeling_df.fillna(0)
    combined_df['principal_comp_1'] = feature_df_pca_2[:,0]
    combined_df['principal_comp_2'] = feature_df_pca_2[:,1]
    return combined_df


def densityBasedClustering(X):
    """Compute DBSCAN."""
    db = DBSCAN(eps=0.03, min_samples=7).fit(X)
    labels = db.labels_
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    plt.figure(figsize = (10, 7))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = labels == k
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=10,
        )
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=4,
        )
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.savefig('DBSCAN.png')
    plt.close()
