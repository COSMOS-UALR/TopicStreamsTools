import numpy as np
import os
import pandas as pd

from ..dataManager import getOutputDir, load_df, save_df
from .input import getChannelData
from .output import outputPeaks, outputConfidenceScoreGraph, outputFrequencyGraph
from .analysis import create_rolling_window_df, extract_anomaly_list, train, data_pre_processing, aggregate_anomalies, transform_anomaly_output

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
            data = getChannelData(settings, channel_id)
            combined_anomalies_list.append(self.getCombinedAnomalies(data, channel_id))
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


    def getAnomaly(self, data, threshold, start_date, anomaly_type, channel_id, columns):
        """For a single type of anomaly dimension, return its findings TODO <needs more description>."""
        settings = self.settings
        x = data.columns.values[1]
        y = data.columns.values[2]
        df = create_rolling_window_df(data, x, y)
        end_date = df.columns.values[1]
        loss_df = self.getLossDF(anomaly_type, channel_id, df.drop([end_date, x, y], axis=1))
        outputFrequencyGraph(settings, loss_df, channel_id, anomaly_type, start_date)
        outputConfidenceScoreGraph(settings, loss_df, channel_id, anomaly_type, start_date)
        outputPeaks(settings, loss_df, channel_id, anomaly_type)
        anomaly_df = extract_anomaly_list(df, loss_df, threshold, start_date)
        aggregated_anomalies = aggregate_anomalies(anomaly_df, x, y)
        out_df = aggregated_anomalies.copy()
        out_df = pd.DataFrame(data=[
                    [pd.to_datetime(df['date'][0]),
                     pd.to_datetime(df['end_date'][(df.shape[0] - 1)]),
                     str(pd.to_datetime(df['end_date'][(df.shape[0] - 1)]) - pd.to_datetime(df['date'][0]))
                         .replace(' 00:00:00', ""), '0', '0', float(0), float(0), float(0)
                    ]
                ], columns=columns)
        return transform_anomaly_output(out_df, anomaly_type, channel_id)


    def getLossDF(self, anomaly_type, channel_id, df):
        """For the given channel_id's df, will attempt to load the loss dataframe, or will retrain a model."""
        settings = self.settings
        loss_df_file = f"{channel_id}_{anomaly_type}.pkl"
        try:
            loss_df = load_df(settings, loss_df_file)
        except FileNotFoundError:
            loss_df = None
        if loss_df is None or ('retrain' in settings and settings['retrain']):
            X, Y, T = data_pre_processing(df, settings['lookback_size'])
            loss = train(X, Y, settings['model'], anomaly_type)
            # Adjust index and date column for later plotting
            loss_df = pd.DataFrame(loss, columns=["loss"], index=pd.to_datetime(T))
            loss_df["date"] = pd.to_datetime(T)
            save_df(settings, loss_df_file, loss_df)
        return loss_df


    def getCombinedAnomalies(self, data, channel_id):
        """Generate anomaly dataframe for each dimension and return combined dataframe."""
        settings = self.settings
        threshold = settings['threshold']
        start_date = settings['filters']['in']['start_date'] if 'start_date' in settings['filters']['in'] else None
        # Views subs
        columns = ['start_date', 'end_date', 'duration', 'avg_views', 'avg_subscribers', 'avg_corr', 'avg_anomaly_score', 'avg_sse']
        views_subs_data = data[['date', 'total_views', 'total_subscribers']]
        views_subs_transformed_anomalies = self.getAnomaly(views_subs_data, threshold, start_date, 'views_subs', channel_id, columns)
        # Views videos
        columns = ['start_date', 'end_date', 'duration', 'avg_views', 'avg_videos', 'avg_corr', 'avg_anomaly_score', 'avg_sse']
        views_videos_data = data[['date', 'total_views', 'total_videos']]
        views_videos_transformed_anomalies = self.getAnomaly(views_videos_data, threshold, start_date, 'views_videos', channel_id, columns)
        # Views comments
        columns = ['start_date', 'end_date', 'duration', 'avg_views', 'avg_comments', 'avg_corr', 'avg_anomaly_score', 'avg_sse']
        views_comments_data = data[['date', 'total_views', 'total_comments']]
        views_comments_transformed_anomalies = self.getAnomaly(views_comments_data, threshold, start_date, 'views_comments', channel_id, columns)
        # Subs videos
        columns = ['start_date', 'end_date', 'duration', 'avg_subscribers', 'avg_videos', 'avg_corr', 'avg_anomaly_score', 'avg_sse']
        subs_videos_data = data[['date', 'total_subscribers', 'total_videos']]
        subs_videos_transformed_anomalies = self.getAnomaly(subs_videos_data, threshold, start_date, 'subs_videos', channel_id, columns)
        # Subs comments
        columns = ['start_date', 'end_date', 'duration', 'avg_subscribers', 'avg_comments', 'avg_corr', 'avg_anomaly_score', 'avg_sse']
        subs_comments_data = data[['date', 'total_subscribers', 'total_comments']]
        subs_comments_transformed_anomalies = self.getAnomaly(subs_comments_data, threshold, start_date, 'subs_comments', channel_id, columns)
        # Video comments
        columns = ['start_date', 'end_date', 'duration', 'avg_videos', 'avg_comments', 'avg_corr', 'avg_anomaly_score', 'avg_sse']
        videos_comments_data = data[['date', 'total_videos', 'total_comments']]
        videos_comments_transformed_anomalies = self.getAnomaly(videos_comments_data, threshold, start_date, 'videos_comments', channel_id, columns)
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
