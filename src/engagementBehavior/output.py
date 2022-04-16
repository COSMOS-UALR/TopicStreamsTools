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
