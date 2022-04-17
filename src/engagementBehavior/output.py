import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from ..dataManager import getOutputDir

sns.set_context('talk', font_scale=.8)

def outputFrequencyGraph(settings, loss_df, channel_id, anomaly_type, start_date):
    """Graph loss frequency."""
    date_output = start_date.strftime('%Y-%m-%d') if start_date else ''
    ax = sns.displot(data=loss_df, x="loss", kde=True, bins=100, label="Frequency", height=10, aspect=2)
    ax.set(xlabel='Anomaly Confidence Score',
        ylabel='Frequency (sample)',
        title='Frequency Distribution | Kernel Density Estimation')
    # plt.axvline(1.80, color="k", linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(getOutputDir(settings), f'{channel_id}_{anomaly_type}_{date_output}_Frequency_Distribution.png'))
    plt.close()


def outputLossGraph(settings, loss_df, peaks, channel_id, anomaly_type, start_date):
    """Graph losses over time and highlight peaks."""
    date_output = start_date.strftime('%Y-%m-%d') if start_date else ''
    values = np.array(loss_df['loss'])
    peak_dates = list(loss_df.iloc[peaks]['date'])
    peak_values = list(values[peaks])
    plt.figure(figsize=(20, 10))
    ax = sns.scatterplot(x=peak_dates, y=peak_values, marker='x', color='r')
    ax = sns.lineplot(x="date", y="loss", data=loss_df, color='g', label="Anomaly Score")
    ax.set_title("Anomaly Confidence Score vs Timestamp")
    ax.set(ylabel="Anomaly Confidence Score", xlabel="Date")
    plt.savefig(os.path.join(getOutputDir(settings), f'{channel_id}_{anomaly_type}_{date_output}_Anomaly_Confidence_Score_With_Peaks.png'))
    plt.close()


def hierarchicalClustering(labeling_df):
    # plt.style.use('ggplot')
    cols = ['channel_id', 'start_date', 'end_date', 'duration(in days)', 'max_anomaly_score(views_subs)',
        'min_corr(views_subs)','max_anomaly_score(views_videos)', 'min_corr(views_videos)',
        'max_anomaly_score(views_comments)','min_corr(views_comments)','max_anomaly_score(subs_videos)', 'min_corr(subs_videos)',
        'max_anomaly_score(subs_comments)', 'min_corr(subs_comments)','max_anomaly_score(videos_comments)', 'min_corr(videos_comments)']
    clustering_df = labeling_df[cols].copy()
    clustering_df.fillna(0, inplace=True)
    clustering_df.set_index('channel_id', inplace=True)
    clustering_df.drop(columns=['start_date', 'end_date', 'duration(in days)'], axis=1, inplace=True)
    sns.clustermap(clustering_df, cmap='mako', figsize=(20,20))
    plt.savefig(f'engagement_stats_clustermap.png')
    plt.close()


def outputSuspicion(all_rankings):
    rankings = ['suspicion rank - Max Rank',
        'suspicion rank - Sum Rank',
        'suspicion rank - Mean Rank',
        'suspicion rank - Harmonic Rank']
    methods = ['pearson', 'spearman', 'kendall']
    all_rankings = all_rankings[rankings]
    for method in methods:
        fig=plt.gcf()
        fig.set_size_inches(10,10)
        fig=sns.heatmap(all_rankings.corr(method=method),annot=True,linewidths=1,linecolor='k',
                        square=True,mask=False, vmin=-1, vmax=1,
                        cbar_kws={"orientation": "vertical"},cbar=True)
        plt.savefig(f'{method}_heatmap.png')
        plt.close()


def outputVariance(pca_12):
    print("Variance explained by all 12 principal components = ", sum(pca_12.explained_variance_ratio_ * 100))
    plt.plot(np.cumsum(pca_12.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Explained Variance')
    plt.savefig('Explained Variance.png')
    plt.close()
    print("Variance explained by the First 2 principal components: ", np.cumsum(pca_12.explained_variance_ratio_ *100)[2])
    print("Variance explained by the First 3 principal components: ", np.cumsum(pca_12.explained_variance_ratio_ *100)[3])
    print("Variance explained by the First 4 principal components: ", np.cumsum(pca_12.explained_variance_ratio_ *100)[4])
    print("Variance explained by the First 5 principal components: ", np.cumsum(pca_12.explained_variance_ratio_ *100)[5])
    print("Variance explained by the First 6 principal components: ", np.cumsum(pca_12.explained_variance_ratio_ *100)[6])


def outputScatterPlot(feature_df_pca_2, pca_12, suspicion_df):
    """Create a scatterplot of the data using the values of the two principal components."""
    plt.figure(figsize =(10, 7))
    sns.scatterplot(x=feature_df_pca_2[:, 0], y=feature_df_pca_2[:, 1],  s=70, hue=suspicion_df)
    sns.color_palette("flare", as_cmap=True)
    plt.title("2D Scatterplot: " + str(np.round(np.cumsum(pca_12.explained_variance_ratio_ *100)[2],2)) + "% of the variability captured", pad=15)
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.savefig('PCA_Plot.png')
    plt.close()
