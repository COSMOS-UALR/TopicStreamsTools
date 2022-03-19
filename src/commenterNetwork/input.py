import errno
import humanize
import networkx as nx
import os
import pandas as pd
from tqdm import tqdm

from ..dataManager import fileExists, get_connection, getFilePath, read_file


def loadNetwork(settings, file):
    file_path = getFilePath(settings, file)
    if not fileExists(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    G = nx.read_gexf('network.gexf')
    return G


def getComments(settings):
    """Return channel data in analysis-ready format."""
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


def quoteList(str_list):
    """Return list values as quoted strings."""
    return ["'" + item + "'" for item in str_list]


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


def countComments(db_connector, comments_table, video_ids):
    count_query = f'SELECT COUNT(comment_id) FROM {comments_table} WHERE video_id IN ({",".join(quoteList(video_ids))})'
    cur = db_connector.cursor()
    cur.execute(count_query)
    return cur.fetchall()[-1][-1]


def queryComments(db_connector, video_ids):
    """Return comment publication counts grouped by day."""
    CHUNKSIZE = 1000
    comments_table = 'comments'
    columns = ['comment_id', 'commenter_name', 'commenter_id', 'comment_displayed', 'comment_original', 'likes', 'total_replies', 'published_date', 'updated_date', 'reply_to', 'video_id']
    query = f'SELECT {",".join(columns)} FROM {comments_table} WHERE video_id IN ({",".join(quoteList(video_ids))})'
    query += ' LIMIT 100000'
    comments_df = None
    chunks = pd.read_sql(query, db_connector, columns=columns, chunksize=CHUNKSIZE)
    total_comments = countComments(db_connector, comments_table, video_ids)
    progress = tqdm(total=total_comments)
    for df in chunks:
        comments_df = pd.concat([comments_df, df])
        progress.set_description(desc=f"Collecting comments : [Last packet size: {humanize.intcomma(df.shape[0])} / Total Comments : {humanize.intcomma(total_comments)}]", refresh=True)
        progress.update(CHUNKSIZE)
    return comments_df
