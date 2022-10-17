import errno
from gensim import models
import os
import pickle as pkl

from ..dataManager import fileExists, getFilePath, get_connection, fetchData, quoteList, countQueryItems


def loadModel(settings, file):
    file_path = getFilePath(settings, file)
    if not fileExists(file_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    if settings['model'] == 'LDA':
        model = models.LdaModel.load(file_path)
    if settings['model'] == 'HDP':
        model = models.HdpModel.load(file_path)
    if settings['model'] == 'LDA-Mallet':
        model = models.wrappers.LdaMallet.load(file_path)
    if settings['model'] == 'Biterm':
        with open(file_path, "rb") as file:
            model = pkl.load(file)
    if settings['model'] != 'Biterm':
        settings['numberTopics'] = model.num_topics
    return model


def queryChannelData(settings, channel_ids):
    """Return video data belonging to channel(s)."""
    db_connector = get_connection(settings['filters']['in']['db_settings'])
    CHUNKSIZE = 10000
    video_table = 'videos'
    columns = ['video_id', 'published_date', 'video_title']
    query_filter = f'FROM {video_table} WHERE channel_id IN ({",".join(quoteList(channel_ids))})'
    # query_filter += f' AND published_date >= "2019-01-01" ORDER BY published_date ASC'
    query = f'SELECT {",".join(columns)} ' + query_filter
    query_count = f'SELECT COUNT(video_id) ' + query_filter
    total_videos = countQueryItems(db_connector, query_count)
    video_df = fetchData(db_connector, query, video_table, columns, CHUNKSIZE, total=total_videos)
    return video_df
