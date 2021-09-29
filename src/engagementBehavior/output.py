import matplotlib.pyplot as plt
import os
import seaborn as sns

from ..dataManager import getOutputDir

sns.set_context('talk', font_scale=.8)

def outputFrequencyGraph(settings, loss_df, channel_id, anomaly_type, start_date):
    if 'folder' in settings['filters']['out']:
        plt.figure(figsize=(20, 10))
        sns.set_style("darkgrid")
        ax = sns.distplot(loss_df["loss"], bins=100, label="Frequency")
        ax.set_title("Frequency Distribution | Kernel Density Estimation")
        ax.set(xlabel='Anomaly Confidence Score', ylabel='Frequency (sample)')
        plt.axvline(1.80, color="k", linestyle="--")
        plt.legend()
        plt.savefig(
            os.path.join(getOutputDir(settings), f'{channel_id}_{anomaly_type}_{start_date}_Frequency_Distribution.png'))


def outputConfidenceScoreGraph(settings, loss_df, channel_id, anomaly_type, start_date):
    if 'folder' in settings['filters']['out']:
        plt.figure(figsize=(20, 10))
        ax = sns.lineplot(x="date", y="loss", data=loss_df, color='g', label="Anomaly Score")
        ax.set_title("Anomaly Confidence Score vs Timestamp")
        ax.set(ylabel="Anomaly Confidence Score", xlabel="Date")
        plt.legend()
        plt.savefig(
            os.path.join(getOutputDir(settings), f'{channel_id}_{anomaly_type}_{start_date}Anomaly_Confidence_Score.png'))
