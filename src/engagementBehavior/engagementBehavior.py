import numpy as np
import os
import pandas as pd

from .analysis import get_anomalies
from .input import readData


class EngagementBehaviorNode:

    def __init__(self, project_name, settings) -> None:
        self.DISTRIB_FILE = 'distribDataframe.pkl'
        self.IDS_FILE = 'ids.pkl'
        node = list(settings.keys())[0]
        self.settings = settings[node]
        self.settings['node'] = node
        self.settings['datasetName'] = project_name

    def run(self):
        settings = self.settings
        print(f"BEGIN {settings['node']}")
        self.getAnomaliesList()
        # if 'file' in settings['filters']['in']:
        #     pass
        # if 'node' in settings['filters']['in']:
        #     pass
        # if 'folder' in settings['filters']['out']:
        #     pass
        # if 'node' in settings['filters']['out']:
        #     pass
        print(f"NODE {settings['node']} END")

    def getAnomaliesList(self):
        combined_anomalies_list = []
        combined_anomalies_list.append(self.getCombinedAnomalies(self.settings['channel_id']))
        # file_list = []
        # file_path = '/content/drive/MyDrive/datasets/'
        # for root, dirs, files in os.walk(file_path):
        #     for file in files:
        #         file_list.append((os.path.join(root, file), file.split('.')[0]))
        # print(file_list)
        # for file in file_list:
        #     combined_anomalies_list.append(self.getCombinedAnomalies(file))
        combined_dfs = pd.concat(combined_anomalies_list, axis=0)
        combined_dfs.reset_index(inplace=True, drop=True)
        combined_dfs = combined_dfs.replace(np.NaN, 0)
        print(combined_dfs.head())
        combined_dfs.to_csv('combined_anomalies.csv', index=False)

    def getCombinedAnomalies(self, channel_id):
        settings = self.settings
        trunc_data = readData(settings)
        threshold = settings['threshold']
        start_date = settings['start_date']

        columns = ['start_date', 'end_date', 'duration', 'avg_views', 'avg_subscribers', 'avg_corr',
                   'avg_anomaly_score', 'avg_sse']
        data = trunc_data[['date', 'daily_views', 'daily_subscribers', 'total_views', 'total_subscribers']]
        views_subs_transformed_anomalies = get_anomalies(settings, data, threshold, start_date, 'views_subs',
                                                         channel_id, columns)

        columns = ['start_date', 'end_date', 'duration', 'avg_views', 'avg_videos', 'avg_corr', 'avg_anomaly_score',
                   'avg_sse']
        data = trunc_data[['date', 'daily_views', 'daily_videos', 'total_views', 'total_videos']]
        views_videos_transformed_anomalies = get_anomalies(settings, data, threshold, start_date, 'views_videos',
                                                           channel_id, columns)

        columns = ['start_date', 'end_date', 'duration', 'avg_views', 'avg_comments', 'avg_corr', 'avg_anomaly_score',
                   'avg_sse']
        data = trunc_data[['date', 'daily_views', 'daily_comments', 'total_views', 'total_comments']]
        views_comments_transformed_anomalies = get_anomalies(settings, data, threshold, start_date, 'views_comments',
                                                             channel_id, columns)

        columns = ['start_date', 'end_date', 'duration', 'avg_subscribers', 'avg_videos', 'avg_corr',
                   'avg_anomaly_score', 'avg_sse']
        data = trunc_data[['date', 'daily_subscribers', 'daily_videos', 'total_subscribers', 'total_videos']]
        subs_videos_transformed_anomalies = get_anomalies(settings, data, threshold, start_date, 'subs_videos',
                                                          channel_id, columns)

        columns = ['start_date', 'end_date', 'duration', 'avg_subscribers', 'avg_comments', 'avg_corr',
                   'avg_anomaly_score', 'avg_sse']
        data = trunc_data[['date', 'daily_subscribers', 'daily_comments', 'total_subscribers', 'total_comments']]
        subs_comments_transformed_anomalies = get_anomalies(settings, data, threshold, start_date, 'subs_comments',
                                                            channel_id, columns)

        columns = ['start_date', 'end_date', 'duration', 'avg_videos', 'avg_comments', 'avg_corr', 'avg_anomaly_score',
                   'avg_sse']
        data = trunc_data[['date', 'daily_videos', 'daily_comments', 'total_videos', 'total_comments']]
        videos_comments_transformed_anomalies = get_anomalies(settings, data, threshold, start_date, 'videos_comments',
                                                              channel_id, columns)

        fields = ['channel_id', 'start_date', 'end_date', 'duration(in days)']
        combined_anomalies = views_subs_transformed_anomalies \
            .merge(views_videos_transformed_anomalies, how='outer', on=fields) \
            .merge(views_comments_transformed_anomalies, how='outer', on=fields) \
            .merge(subs_videos_transformed_anomalies, how='outer', on=fields) \
            .merge(subs_comments_transformed_anomalies, how='outer', on=fields) \
            .merge(videos_comments_transformed_anomalies, how='outer', on=fields)
        combined_anomalies.fillna(0, inplace=True)
        return combined_anomalies
