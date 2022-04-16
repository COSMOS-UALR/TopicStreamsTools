from datetime import timedelta
from functools import partial
import numpy as np
import os
import pandas as pd

from ..dataManager import getOutputDir
from .input import getChannelData
from .output import outputLossGraph, outputFrequencyGraph
from .analysis import create_rolling_window_df, getAnomalyDF, getLossDF, buildAnomalyStats, transform_anomaly_output, getPeaks

class EngagementBehaviorNode:

    def __init__(self, project_name, settings) -> None:
        node = list(settings.keys())[0]
        self.settings = settings[node]
        self.settings['node'] = node
        self.settings['datasetName'] = project_name

    def run(self, previous_node_output):
        settings = self.settings
        print(f"BEGIN {settings['node']}")
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
        if 'folder' in settings['filters']['out']:
            out = os.path.join(getOutputDir(settings), 'combined_anomalies.csv')
            combined_dfs.to_csv(out, index=False)
        # if 'node' in settings['filters']['out']:
        #     TODO
        print(f"NODE {settings['node']} END")

    
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
        """For a timestamped two dimensional array, compute correlations and train model to find anomalies. Output anomalies graphs and return aggregated dataframe on given timeframe."""
        settings = self.settings
        x = data.columns.values[1]
        y = data.columns.values[2]
        df = create_rolling_window_df(data, x, y)
        end_date = df.columns.values[1]
        loss_df = getLossDF(settings, anomaly_type, channel_id, df.drop([end_date, x, y], axis=1))
        peaks, peaks_df = getPeaks(loss_df, channel_id)
        outputFrequencyGraph(settings, loss_df, channel_id, anomaly_type, start_date)
        outputLossGraph(settings, loss_df, peaks, channel_id, anomaly_type, start_date)
        anomaly_df = getAnomalyDF(df, loss_df, start_date)
        aggregated_anomalies = buildAnomalyStats(anomaly_df, x, y, anomaly_aggregation_timeframe)
        return transform_anomaly_output(aggregated_anomalies, anomaly_type, channel_id)


    def getCombinedAnomalies(self, data, channel_id, anomaly_aggregation_timeframe):
        """Generate anomaly dataframe for each dimension and return combined dataframe."""
        settings = self.settings
        start_date = pd.to_datetime(settings['filters']['in']['start_date']) if 'start_date' in settings['filters']['in'] else None
        callAnomalyFunc = partial(self.getAnomaly, start_date, channel_id, anomaly_aggregation_timeframe)
        # Views subs
        views_subs_data = data[['date', 'total_views', 'total_subscribers']]
        views_subs_transformed_anomalies = callAnomalyFunc(views_subs_data, 'views_subs')
        # Views videos
        views_videos_data = data[['date', 'total_views', 'total_videos']]
        views_videos_transformed_anomalies = callAnomalyFunc(views_videos_data, 'views_videos')
        # Views comments
        views_comments_data = data[['date', 'total_views', 'total_comments']]
        views_comments_transformed_anomalies = callAnomalyFunc(views_comments_data, 'views_comments')
        # Subs videos
        subs_videos_data = data[['date', 'total_subscribers', 'total_videos']]
        subs_videos_transformed_anomalies = callAnomalyFunc(subs_videos_data, 'subs_videos')
        # Subs comments
        subs_comments_data = data[['date', 'total_subscribers', 'total_comments']]
        subs_comments_transformed_anomalies = callAnomalyFunc(subs_comments_data, 'subs_comments')
        # Video comments
        videos_comments_data = data[['date', 'total_videos', 'total_comments']]
        videos_comments_transformed_anomalies = callAnomalyFunc(videos_comments_data, 'videos_comments')
        # Combination
        fields = ['channel_id', 'start_date', 'end_date', 'duration(in days)']
        combined_anomalies = views_subs_transformed_anomalies \
            .merge(views_videos_transformed_anomalies, how='outer', on=fields) \
            .merge(views_comments_transformed_anomalies, how='outer', on=fields) \
            .merge(subs_videos_transformed_anomalies, how='outer', on=fields) \
            .merge(subs_comments_transformed_anomalies, how='outer', on=fields) \
            .merge(videos_comments_transformed_anomalies, how='outer', on=fields)
        combined_anomalies.fillna(0, inplace=True)
        return combined_anomalies
