import pandas as pd
import numpy as np
from statistics import harmonic_mean

from .output import hierarchicalClustering, outputSuspicion


METHODS = {
    'Max': max,
    'Sum': sum,
    'Mean': np.mean,
    'Harmonic': harmonic_mean,
}


FEATURE_PAIRS = {
    'views_subs': ['total_views', 'total_subscribers'],
    'views_videos': ['total_views', 'total_videos'],
    'views_comments': ['total_views', 'total_comments'],
    'subs_videos': ['total_subscribers', 'total_videos'],
    'subs_comments': ['total_subscribers', 'total_comments'],
    'videos_comments': ['total_videos', 'total_comments'],
}


def getCombinedSuspicionRank(df):
    """Input: combined_anomalies_df"""
    hierarchicalClustering(df)
    summarize(df)
    # Weighted Combination of the Six Pairwise Features
    # Using the data summary above, we use the proportions to assign weights to each pairwise feature
    df.fillna(0, inplace=True)
    for anomaly_type, _ in FEATURE_PAIRS.items():
        df[f'suspicion_rank({anomaly_type})'] = get_suspicion_rank(df[f'max_anomaly_score({anomaly_type})'], df[f'min_corr({anomaly_type})'])
    # Correlation Between Ranking Methods
    join_fields = ['channel_id', 'start_date', 'end_date', 'duration(in days)']
    combined_rankings = None
    for method, func in METHODS.items():
        ranking_df = applyStatToDF(df, method, func)
        if combined_rankings is None:
            combined_rankings = ranking_df
        else:
            combined_rankings = combined_rankings.merge(ranking_df, how='outer', on=join_fields)
    outputSuspicion(combined_rankings)
    combined_rankings.to_csv(f'Weighted_Rankings.csv', index=False)
    return df, combined_rankings


def get_suspicion_rank(anomaly_score, min_corr):
    suspicion_list = []
    for i in range(anomaly_score.shape[0]):
        # if anomaly_score.loc[i] < 0.5:
        if anomaly_score.loc[i] < 0.2:
            suspicion_list.append(1)
        else:
            if min_corr.loc[i] >= 0.5 and min_corr.loc[i] <= 1:
                suspicion_list.append(2)
            elif min_corr.loc[i] >= 0 and min_corr.loc[i] < 0.5:
                suspicion_list.append(3)
            elif min_corr.loc[i] < 0 and min_corr.loc[i] >= -0.5:
                suspicion_list.append(4)  
            else:
                suspicion_list.append(5)
    return suspicion_list


def summarize(labeling_df):
    fields = ['channel_id', 'start_date', 'end_date', 'duration(in days)']
    summary_df = None
    for anomaly_type, _ in FEATURE_PAIRS.items():
        print(f'Beginning analysis on the pair: {anomaly_type}.')
        labels = labeling_df[fields + [f'max_anomaly_score({anomaly_type})', f'min_corr({anomaly_type})']]
        max_anomaly_score = labels[f'max_anomaly_score({anomaly_type})']
        min_corr = labels[f'min_corr({anomaly_type})']
        labels.dropna(axis=0, inplace=True)
        labels.reset_index(drop=True, inplace=True)
        labels[f'suspicion_rank({anomaly_type})'] = get_suspicion_rank(max_anomaly_score, min_corr)
        # labels.to_csv(f'{anomaly_type}_labels.csv', index=False)
        DDDDDD = pd.DataFrame(data=[
            [anomaly_type,
            (labels[f'suspicion_rank({anomaly_type})'] == 1).sum(),
            (labels[f'suspicion_rank({anomaly_type})'] == 2).sum(),
            (labels[f'suspicion_rank({anomaly_type})'] == 3).sum(),
            (labels[f'suspicion_rank({anomaly_type})'] == 4).sum(),
            (labels[f'suspicion_rank({anomaly_type})'] == 5).sum()],
        ], columns=['Feature', 'Rank 1', 'Rank 2', 'Rank 3', 'Rank 4', 'Rank 5'])
        if summary_df is None:
            summary_df = DDDDDD
        else:
            summary_df = pd.concat((summary_df, DDDDDD), axis = 0)
    summary_df.to_csv(f'feature_summaries.csv', index=False)


def getWeightedInput(df, i):
    return [
        df['suspicion_rank(views_subs)'][i] * 0.35,
        df['suspicion_rank(videos_comments)'][i] * 0.25,
        df['suspicion_rank(subs_comments)'][i] * 0.2,
        df['suspicion_rank(views_videos)'][i] * 0.1,
        df['suspicion_rank(views_comments)'][i] * 0.06,
        df['suspicion_rank(subs_videos)'][i] * 0.04,
    ]


def applyStatToDF(labeling_df, method, func):
    combined_rank_list = []
    for i in range(labeling_df.shape[0]):
        weighted_input = getWeightedInput(labeling_df, i)
        combined_rank = round(func(weighted_input), 2)
        combined_rank_list.append(combined_rank)
    df = labeling_df.copy()
    df[f'suspicion rank - {method} Rank'] = combined_rank_list
    df[f'suspicion rank - {method} Rank'].sort_values(ascending=False)
    cols = ['channel_id', 'start_date', 'end_date', 'duration(in days)',f'suspicion rank - {method} Rank']
    labeled_df = df[cols]
    return labeled_df
