import errno
import networkx as nx
import os
import pandas as pd

from ..dataManager import fileExists, get_connection, getFilePath, read_file, fetchData, countQueryItems, quoteList


def loadNetwork(settings, file):
    """Load the networkx graph from disk."""
    file_path = getFilePath(settings, file)
    if not fileExists(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    G = nx.read_gexf(file_path)
    return G


def getComments(settings):
    """Fetch and return comments data as source-target relationships."""
    df_comments = None
    if 'file' in settings['filters']['in']:
        file = settings['filters']['in']['file']
        file_path = os.path.join(os.getcwd(), 'Data', file)
        df_comments = read_file(settings, file_path, use_all_columns=True)
    elif 'channel_ids' in settings['filters']['in']:
        for channel_id in settings['filters']['in']['channel_ids']:
            df_comments = pd.concat([df_comments, queryData(settings, channel_id)])
    df1 = df_comments[["commenter_id", "video_id"]]
    df_comments = df1.groupby(['commenter_id', 'video_id']).size().reset_index(name='counts')
    df_comments.columns = ['source', 'target', 'strength']
    df_comments = df_comments[['source', 'target']]
    total_target = df_comments.target.nunique()
    return df_comments


def getChannelFileName(channel_id):
    """Return the name of the temporary channel data file."""
    return f"{channel_id}.pkl"


def queryData(settings, channel_id):
    """Fetches channel data from the DB."""
    db_connector = get_connection(settings['filters']['in']['db_settings'])
    video_ids = queryVideos(db_connector, channel_id)
    if len(video_ids) == 0:
        print("No videos found - skipping comments.")
        return None
    df = queryComments(db_connector, video_ids)
    return df


def queryVideos(db_connector, channel_id):
    """Return video IDs belonging to channel and video publication counts grouped by day."""
    video_table = 'videos'
    columns = ['video_id']
    query = f'SELECT {",".join(columns)} FROM {video_table} WHERE channel_id = "{channel_id}"'
    video_df = pd.read_sql(query, db_connector)
    print(f"Loaded {video_df.shape[0]} videos from channel {channel_id}.")
    return list(video_df['video_id'])


def queryComments(db_connector, video_ids):
    """Return raw comments data available for given video ids."""
    CHUNKSIZE = 1000
    comments_table = 'comments'
    # columns = ['comment_id', 'commenter_name', 'commenter_id', 'comment_displayed', 'comment_original', 'likes', 'total_replies', 'published_date', 'updated_date', 'reply_to', 'video_id']
    columns = ['commenter_id', 'video_id']
    query_filter = f'FROM {comments_table} WHERE video_id IN ({",".join(quoteList(video_ids))})'
    query_count = f'SELECT COUNT(comment_id) ' + query_filter
    query = f'SELECT {",".join(columns)} ' + query_filter
    query += ' LIMIT 100000'
    total_comments = countQueryItems(db_connector, query_count)
    comments_df = fetchData(db_connector, query, comments_table, columns, CHUNKSIZE, total_comments)
    return comments_df
