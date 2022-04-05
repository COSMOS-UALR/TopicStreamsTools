import pandas as pd

from ..dataManager import get_connection, load_df, save_df, fetchData, countQueryItems, quoteList


CHUNKSIZE = 10000

def getChannelData(settings, channel_id, video_ids=None):
    """Return channel data in analysis-ready format."""
    try:
        df = load_df(settings, getChannelFileName(channel_id))
    except FileNotFoundError:
        df = queryChannelData(settings, channel_id, video_ids)
    df = computeTotalValues(df)
    df = computeDailyValues(df)
    return truncateData(df)


def truncateData(df):
    """Truncate channel dataframe to exclude rows with no views."""
    thresh = 0
    non_zero_views_rows = df[df['total_views']>0].index.tolist()
    if len(non_zero_views_rows) > 0:    
        thresh = non_zero_views_rows[0]
    try:
        df = df.truncate(before=thresh, axis=0)
    except:
        index_list = []
        for i in range(df.shape[0]):
            index_list.append(i)
        df['date'] = df.index
        df.index = index_list
        df = df.truncate(before=thresh, axis=0)
    return df.reset_index(drop=True)


def computeTotalValues(df):
    """Adjust values for totals videos and comments. """
    total_videos_list = [df['daily_videos'][0]]
    total_comments_list = [df['daily_comments'][0]]
    for i in range(df.shape[0] - 1):
        total_videos_list.append(total_videos_list[-1])
        total_videos_list[-1] += df['daily_videos'][i + 1]
        total_comments_list.append(total_comments_list[-1])
        total_comments_list[-1] += df['daily_comments'][i + 1]
    df['total_videos'] = total_videos_list
    df['total_comments'] = total_comments_list
    return df


def computeDailyValues(df):
    """Adjust values for daily views and subscribers."""
    daily_views_list = [df['total_views'][0]]
    daily_subscribers_list = [df['total_subscribers'][0]]
    for i in range(1, df.shape[0]):
        daily_views_list.append(df['total_views'][i])
        daily_views_list[-1] -= df['total_views'][i - 1]
        daily_subscribers_list.append(df['total_subscribers'][i])
        daily_subscribers_list[-1] -= df['total_subscribers'][i - 1]
    df['daily_views'] = daily_views_list
    df['daily_subscribers'] = daily_subscribers_list
    return df


def getChannelFileName(channel_id):
    """Return the name of the temporary channel data file."""
    return f"{channel_id}.pkl"


def queryChannelData(settings, channel_id, video_ids=None):
    """Fetches channel data from the DB."""
    print(f'Starting collection for channe; {channel_id}.')
    table = 'channels_daily'
    columns = ['channel_id', 'total_views', 'total_subscribers', 'total_videos', 'extracted_date']
    query = f'SELECT {",".join(columns)} FROM {table} WHERE channel_id = "{channel_id}"'
    db_connector = get_connection(settings['filters']['in']['db_settings'])
    df = fetchData(db_connector, query, table, columns, CHUNKSIZE, total=None)
    df.rename(columns={'total_videos': 'total_videos_in_db'}, inplace=True)
    video_ids, df = getVideoPublicationHistogram(db_connector, df, channel_id, video_ids)
    df = getCommentPublicationHistogram(db_connector, df, video_ids)
    df.rename(columns={'extracted_date': 'date'}, inplace=True)
    # print(f"Filtering out na values from {df.shape[0]} entries.")
    df.dropna(how='any', thresh=200, axis=1, inplace=True)
    # df.dropna(how='any', axis=0, inplace=True)
    # print(f"{df.shape[0]} entries remain.")
    save_df(settings, getChannelFileName(channel_id), df)
    return df


def getVideoPublicationHistogram(db_connector, channel_df, channel_id, video_ids=None):
    """Return video IDs belonging to channel (or matching video IDs when provided) and video publication counts grouped by day."""
    video_table = 'videos'
    columns = ['video_id', 'published_date']
    query = f'SELECT {",".join(columns)} FROM {video_table} WHERE channel_id = "{channel_id}"'
    if video_ids is not None:
        query += f' AND video_id IN ({",".join(quoteList(video_ids))})'
    video_df = fetchData(db_connector, query, video_table, columns, CHUNKSIZE, total=None)
    video_ids = list(video_df['video_id'])
    if len(video_ids) > 0:
        video_df = video_df.groupby(pd.Grouper(key='published_date', axis=0, freq='D')).count()
        video_df.reset_index(inplace=True)
        video_df.rename(columns={'published_date': 'date', 'video_id': 'daily_videos'}, inplace=True)
        channel_df = pd.merge(channel_df, video_df, how='left', left_on='extracted_date', right_on='date')
        channel_df.drop('date', axis=1, inplace=True)
    else:
        channel_df['daily_videos'] = 0
    return video_ids, channel_df


def getCommentPublicationHistogram(db_connector, channel_df, video_ids):
    """Return comment publication counts grouped by day."""
    comments_table = 'comments'
    columns = ['comment_id', 'published_date']
    query_filter = f'FROM {comments_table} WHERE video_id IN ({",".join(quoteList(video_ids))})'
    #TODO - Filtering when not using filters from previous nodes.
    # query_filter += ' LIMIT 1000'
    query_count = f'SELECT COUNT(comment_id) ' + query_filter
    query = f'SELECT {",".join(columns)} ' + query_filter
    if len(video_ids) > 0:
        total_comments = countQueryItems(db_connector, query_count)
        comments_df = fetchData(db_connector, query, comments_table, columns, CHUNKSIZE, total_comments)
        # Below accounts for comment date being stored as localized strings
        comments_df['published_date'] = pd.to_datetime(comments_df['published_date']).dt.tz_localize(None)
        comments_df = comments_df.groupby(pd.Grouper(key='published_date', axis=0, freq='D')).count()
        comments_df.reset_index(inplace=True)
        comments_df.rename(columns={'published_date': 'date', 'comment_id': 'daily_comments'}, inplace=True)
        channel_df = pd.merge(channel_df, comments_df, how='left', left_on='extracted_date', right_on='date')
        channel_df.drop('date', axis=1, inplace=True)
    else:
        channel_df['daily_comments'] = 0
    return channel_df
