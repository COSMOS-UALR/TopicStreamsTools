import numpy as np
import os
import pandas as pd

from .analysis import getAnomaly
from .input import getChannelData
from ..dataManager import getOutputDir


class EngagementBehaviorNode:

    def __init__(self, project_name, settings) -> None:
        node = list(settings.keys())[0]
        self.settings = settings[node]
        self.settings['node'] = node
        self.settings['datasetName'] = project_name

    def run(self):
        settings = self.settings
        print(f"BEGIN {settings['node']}")
        combined_anomalies_list = []
        channel_ids = self.fetchChannelIDs()
        for channel_id in channel_ids:
            combined_anomalies_list.append(self.getCombinedAnomalies(channel_id))
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


    def getCombinedAnomalies(self, channel_id):
        """Generate anomaly dataframe for each dimension and return combined dataframe."""
        settings = self.settings
        trunc_data = getChannelData(settings, channel_id)
        threshold = settings['threshold']
        start_date = settings['filters']['in']['start_date'] if 'start_date' in settings['filters']['in'] else None
        # Views subs
        columns = ['start_date', 'end_date', 'duration', 'avg_views', 'avg_subscribers', 'avg_corr',
                   'avg_anomaly_score', 'avg_sse']
        data = trunc_data[['date', 'daily_views', 'daily_subscribers', 'total_views', 'total_subscribers']]
        views_subs_transformed_anomalies = getAnomaly(settings, data, threshold, start_date, 'views_subs',
                                                         channel_id, columns)
        # Views videos
        columns = ['start_date', 'end_date', 'duration', 'avg_views', 'avg_videos', 'avg_corr', 'avg_anomaly_score',
                   'avg_sse']
        data = trunc_data[['date', 'daily_views', 'daily_videos', 'total_views', 'total_videos']]
        views_videos_transformed_anomalies = getAnomaly(settings, data, threshold, start_date, 'views_videos',
                                                           channel_id, columns)
        # Views comments
        columns = ['start_date', 'end_date', 'duration', 'avg_views', 'avg_comments', 'avg_corr', 'avg_anomaly_score',
                   'avg_sse']
        data = trunc_data[['date', 'daily_views', 'daily_comments', 'total_views', 'total_comments']]
        views_comments_transformed_anomalies = getAnomaly(settings, data, threshold, start_date, 'views_comments',
                                                             channel_id, columns)
        # Subs videos
        columns = ['start_date', 'end_date', 'duration', 'avg_subscribers', 'avg_videos', 'avg_corr',
                   'avg_anomaly_score', 'avg_sse']
        data = trunc_data[['date', 'daily_subscribers', 'daily_videos', 'total_subscribers', 'total_videos']]
        subs_videos_transformed_anomalies = getAnomaly(settings, data, threshold, start_date, 'subs_videos',
                                                          channel_id, columns)
        # Subs comments
        columns = ['start_date', 'end_date', 'duration', 'avg_subscribers', 'avg_comments', 'avg_corr',
                   'avg_anomaly_score', 'avg_sse']
        data = trunc_data[['date', 'daily_subscribers', 'daily_comments', 'total_subscribers', 'total_comments']]
        subs_comments_transformed_anomalies = getAnomaly(settings, data, threshold, start_date, 'subs_comments',
                                                            channel_id, columns)
        # Video comments
        columns = ['start_date', 'end_date', 'duration', 'avg_videos', 'avg_comments', 'avg_corr', 'avg_anomaly_score',
                   'avg_sse']
        data = trunc_data[['date', 'daily_videos', 'daily_comments', 'total_videos', 'total_comments']]
        videos_comments_transformed_anomalies = getAnomaly(settings, data, threshold, start_date, 'videos_comments',
                                                              channel_id, columns)
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
