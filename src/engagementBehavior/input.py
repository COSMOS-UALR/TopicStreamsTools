import pandas as pd

from ..dataManager import get_connection, load_df, save_df


def getChannelData(settings, channel_id):
    """Return channel data in analysis-ready format."""
    raw_data = load_df(settings, getChannelFileName(channel_id))
    if raw_data is None:
        raw_data = queryChannelData(settings, channel_id)
    total_videos = [raw_data['daily_videos'][0]]
    total_comments = [raw_data['daily_comments'][0]]
    daily_views = [raw_data['total_views'][0]]
    daily_subscribers = [raw_data['total_subscribers'][0]]
    for i in range(raw_data.shape[0] - 1):
        videos_total = total_videos[i] + raw_data['daily_videos'][i + 1]
        comments_total = total_comments[i] + raw_data['daily_comments'][i + 1]
        total_videos.append(videos_total)
        total_comments.append(comments_total)
    for j in range(1, raw_data.shape[0]):
        views_daily = raw_data['total_views'][j] - raw_data['total_views'][j - 1]
        subs_daily = raw_data['total_subscribers'][j] - raw_data['total_subscribers'][j - 1]
        daily_views.append(views_daily)
        daily_subscribers.append(subs_daily)
    raw_data['total_videos'] = total_videos
    raw_data['total_comments'] = total_comments
    raw_data['daily_views'] = daily_views
    raw_data['daily_subscribers'] = daily_subscribers
    for i in range(raw_data.shape[0]):
        if raw_data['total_views'][i] == 0:
            continue
        else:
            thresh = i
        break
    try:
        thresh
    except:
        thresh = 0
    try:
        trunc_data = raw_data.truncate(before=thresh, axis=0)
        trunc_data
    except:
        index_list = []
        for i in range(raw_data.shape[0]):
            index_list.append(i)
        raw_data['date'] = raw_data.index
        raw_data.index = index_list
        trunc_data = raw_data.truncate(before=thresh, axis=0)
    trunc_data.reset_index(inplace=True, drop=True)
    return trunc_data


def quoteList(str_list):
    """Return list values as quoted strings."""
    return ["'" + item + "'" for item in str_list]


def getChannelFileName(channel_id):
    """Return the name of the temporary channel data file."""
    return f"{channel_id}.pkl"


def queryChannelData(settings, channel_id):
    """Fetches channel data from the DB."""
    table = 'channels_daily'
    columns = ['channel_id', 'total_views', 'total_subscribers', 'total_videos', 'extracted_date']
    query = f'SELECT {",".join(columns)} FROM {table} WHERE channel_id = "{channel_id}"'
    db_connector = get_connection(settings['filters']['in']['db_settings'])
    df = pd.read_sql(query, db_connector, columns=columns)
    df.rename(columns={'total_videos': 'total_videos_in_db'}, inplace=True)
    video_ids, df = getVideoPublicationHistogram(db_connector, df, channel_id)
    df = getCommentPublicationHistogram(db_connector, df, video_ids)
    df.rename(columns={'extracted_date': 'date'}, inplace=True)
    # print(f"Filtering out na values from {df.shape[0]} entries.")
    df.dropna(how='any', thresh=200, axis=1, inplace=True)
    # df.dropna(how='any', axis=0, inplace=True)
    # print(f"{df.shape[0]} entries remain.")
    # print(df.head())
    save_df(settings, getChannelFileName(channel_id), df)
    return df


def getVideoPublicationHistogram(db_connector, channel_df, channel_id):
    """Return video IDs belonging to channel and video publication counts grouped by day."""
    video_table = 'videos'
    columns = ['video_id', 'published_date']
    query = f'SELECT {",".join(columns)} FROM {video_table} WHERE channel_id = "{channel_id}"'
    video_df = pd.read_sql(query, db_connector)
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
    date_columns = {'published_date': {'format': r'%Y-%m-%dT %H:%M:%SZ'}}
    query = f'SELECT {",".join(columns)} FROM {comments_table} WHERE video_id IN ({",".join(quoteList(video_ids))})'
    if len(video_ids) > 0:
        comments_df = pd.read_sql(query, db_connector, parse_dates=date_columns)
        # Below accounts for comment date being stored as localized strings
        comments_df['published_date'] = pd.to_datetime(comments_df['published_date']).dt.tz_localize(None)
        comments_df = comments_df.groupby(pd.Grouper(key='published_date', axis=0, freq='D')).count()
        comments_df.reset_index(inplace=True)
        comments_df.rename(columns={'published_date': 'date', 'comment_id': 'daily_comments'}, inplace=True)
        df = pd.merge(channel_df, comments_df, how='left', left_on='extracted_date', right_on='date')
        df.drop('date', axis=1, inplace=True)
    else:
        channel_df['daily_comments'] = 0
    return channel_df