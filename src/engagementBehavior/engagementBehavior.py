from datetime import timedelta
from functools import partial
import numpy as np
import os
import pandas as pd

from ..dataManager import getOutputDir, load_df, save_df
from .input import getChannelData
from .output import outputLossGraph, outputFrequencyGraph
from .analysis import create_rolling_window_df, getAnomalyDF, getLossDF, buildAnomalyStats,\
    transform_anomaly_output, getPeaks, getPeakIntensityDf
from .suspicion import getCombinedSuspicionRank
from .PCA import applyPCA


class EngagementBehaviorNode:

    def __init__(self, project_name, settings) -> None:
        node = list(settings.keys())[0]
        self.settings = settings[node]
        self.settings['node'] = node
        self.settings['datasetName'] = project_name
        self.feature_pairs = {
            'views_subs': ['total_views', 'total_subscribers'],
            'views_videos': ['total_views', 'total_videos'],
            'views_comments': ['total_views', 'total_comments'],
            'subs_videos': ['total_subscribers', 'total_videos'],
            'subs_comments': ['total_subscribers', 'total_comments'],
            'videos_comments': ['total_videos', 'total_comments'],
        }

    def run(self, previous_node_output):
        settings = self.settings
        out = {}
        combined_dfs_file = "combined_anomalies.pkl"
        print(f"BEGIN {settings['node']}")
        try:
            combined_dfs = load_df(settings, combined_dfs_file)
        except FileNotFoundError:
            combined_dfs = None
        if combined_dfs is None or ('retrain' in settings and settings['retrain']):
            combined_anomalies_list = []
            anomaly_aggregation_timeframe = timedelta(days=100)
            video_ids = previous_node_output['video_ids'] if 'video_ids' in previous_node_output else None
            channel_ids = self.fetchChannelIDs()
            for channel_id in channel_ids:
                data = getChannelData(settings, channel_id, video_ids)
                combined_anomalies_list.append(self.getCombinedAnomalies(data, channel_id, anomaly_aggregation_timeframe))
            combined_dfs = pd.concat(combined_anomalies_list, axis=0)
            combined_dfs.reset_index(inplace=True, drop=True)
            combined_dfs = combined_dfs.replace(np.NaN, 0)
            save_df(settings, combined_dfs_file, combined_dfs)
            if 'folder' in settings['filters']['out']:
                out = os.path.join(getOutputDir(settings), 'combined_anomalies.csv')
                combined_dfs.to_csv(out, index=False)
        labeling_df, combined_rankings = getCombinedSuspicionRank(combined_dfs)
        applyPCA(labeling_df, combined_rankings)
        # if 'node' in settings['filters']['out']:
        #     TODO
        print(f"NODE {settings['node']} END")
        return out

    
    def fetchChannelIDs(self):
        """Fetches the IDs of the desired channels, depending on input method."""
        channel_ids = []
        if 'channel_id' in self.settings['filters']['in']:
            channel_ids = [self.settings['filters']['in']['channel_id']]
        if 'channel_ids' in self.settings['filters']['in']:
            channel_ids = self.settings['filters']['in']['channel_ids']
        # if 'node' in settings['filters']['in']:
        #     TODO
        return channel_ids


    def getAnomaly(self, start_date, channel_id, anomaly_aggregation_timeframe, data, anomaly_type):
        """For a timestamped two dimensional array, compute correlations and train model to find anomalies. Return loss dataframe and aggregated dataframe on given timeframe."""
        settings = self.settings
        x = data.columns.values[1]
        y = data.columns.values[2]
        df = create_rolling_window_df(data, x, y)
        end_date = df.columns.values[1]
        loss_df = getLossDF(settings, anomaly_type, channel_id, df.drop([end_date, x, y], axis=1))
        anomaly_df = getAnomalyDF(df, loss_df, start_date)
        aggregated_anomalies = buildAnomalyStats(anomaly_df, x, y, anomaly_aggregation_timeframe)
        return loss_df, transform_anomaly_output(aggregated_anomalies, anomaly_type, channel_id)


    def getCombinedAnomalies(self, data, channel_id, anomaly_aggregation_timeframe):
        """Generate anomaly dataframe for each dimension and return combined dataframe."""
        settings = self.settings
        start_date = pd.to_datetime(settings['filters']['in']['start_date']) if 'start_date' in settings['filters']['in'] else None
        callAnomalyFunc = partial(self.getAnomaly, start_date, channel_id, anomaly_aggregation_timeframe)
        combined_anomalies = None
        aggregate_peak_df = None
        aggregate_intensities_df = None
        join_fields = ['channel_id', 'start_date', 'end_date', 'duration(in days)']
        for anomaly_type, feature_pair in self.feature_pairs.items():
            print(f'Beginning analysis on the pair: {anomaly_type}.')
            input = data[['date'] + feature_pair]
            loss_df, anomalies_df = callAnomalyFunc(input, anomaly_type)
            peaks, peaks_df = getPeaks(loss_df, channel_id)
            outputFrequencyGraph(settings, loss_df, channel_id, anomaly_type, start_date)
            outputLossGraph(settings, loss_df, peaks, channel_id, anomaly_type, start_date)
            intensities_df = getPeakIntensityDf(loss_df, peaks)
            if combined_anomalies is None:
                combined_anomalies = anomalies_df
            else:
                combined_anomalies = combined_anomalies.merge(anomalies_df, how='outer', on=join_fields)
            if aggregate_peak_df is None:
                aggregate_peak_df = peaks_df
            else:
                aggregate_peak_df = pd.concat((aggregate_peak_df, peaks_df), axis = 0)
            if aggregate_intensities_df is None:
                aggregate_intensities_df = intensities_df
            else:
                aggregate_intensities_df = pd.concat((aggregate_intensities_df, intensities_df), axis = 0)
        print(f"Accross all features, channel {channel_id} found {sum(aggregate_peak_df['peak_count'])} anomaly peaks. Max peak = {max(aggregate_peak_df['max_peak'])}. Average peak = {np.mean(aggregate_peak_df['avg_peak'])}.")
        intensity_counts = aggregate_intensities_df['peak_intensity'].value_counts().rename_axis('intensity_level').reset_index(name='frequency')
        intensity_counts['distribution'] = intensity_counts['frequency'].apply(lambda x: round((x/aggregate_intensities_df.shape[0]) * 100),1)
        intensity_counts.set_index('intensity_level', inplace=True)
        # intensity_counts.index.name = None
        combined_anomalies.fillna(0, inplace=True)
        return combined_anomalies
