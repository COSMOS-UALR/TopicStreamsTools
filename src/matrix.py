#Topic Modelling
from gensim import models

#Misc
import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from .IO import save_df, DISTRIB_FILE, WORDS_FILE

def get_dominant_topics_counts_and_distribution(doc_topics):
    """
    Function calculates the counts and relative frequencies for each document represented in doc_topics. See the
    function, get_document_topic_matrix(), to understand the assumed format of doc_topics.

    :param doc_topics: Document-topic matrix in numpy format. Each row represents a document's topic distribution.

    :return: Two numpy arrays:
        1) dominant_topic_counts: array of integers where the ith element is the number of  documents from doc_topics
        that have topic i as their dominant topic.

        2) dominant_topic_dist: array of floats where the ith element is the number of documents from doc_topics with
        dominant topic i divided by the total number of documents in doc_topics.
    """

    # Topic counts start at 0 depending on num topics (or rows) in document-topic matrix.
    dominant_topic_counts = np.zeros(len(doc_topics[0]))

    for topic_dist in tqdm(doc_topics):

        # order the topic indices from largest to smallest probability. Without the -1, would get smallest to largest.
        sorted_topic_indices = np.argsort(-1 * topic_dist)

        # The topic index with the largest probability (ie, first in the ordered list)
        dominant_topic = sorted_topic_indices[0]

        # Increment the counter for the corresponding topic in dominant_topic_counts.
        dominant_topic_counts[dominant_topic] += 1

    # Normalize counts to get distribution of dominant topics in the input documents.
    dominant_topic_dist = np.divide(dominant_topic_counts, np.sum(dominant_topic_counts), dtype=np.float64)

    # Ensure that the sum of the distribution is pretty close to 1. Checking precision out to 5 decimal places here.
    assert np.around(np.sum(dominant_topic_dist), 5) == 1.0

    return dominant_topic_counts, dominant_topic_dist


def print_dominant_topics_by_frequency(dominant_topic_counts, dominant_topic_dist):
    """
    Function shows a simple way of printing useful information about the most frequent dominant topics.

    Both of the parameters for this function come from get_dominant_topics_counts_and_distribution() above.
    :param dominant_topic_counts: Array of topic counts such that the ith element is the number of documents
    with dominant topic i.

    :param dominant_topic_dist: Normalized form of dominant_topic_counts such that the ith element is the number of
    documents with dominant topic i divided by the number of documents.
    """

    # Get topic indices ordered from most to least frequent dominant topics.
    ordered_topic_indices = np.argsort(-1 * dominant_topic_counts)

    # Print the topic index, the number of documents with the dominant topic, and the proportion of documents.
    for topic_index in ordered_topic_indices:
        print('topic ' + str(topic_index)
              + ', count=' + str(dominant_topic_counts[topic_index])
              + ', proportion=' + str(np.around(dominant_topic_dist[topic_index], 4)))
    print()

def get_doc_topic_distributions(model, corpus):
   topic_distributions = []
   #FILTER BY DATES HERE
   for i, doc in enumerate(tqdm(corpus)):
       doc_dist = model[doc]
       doc_distribution = np.zeros(len(doc_dist), dtype='float64')
       for (topic, val) in doc_dist:
           doc_distribution[topic] = val
       topic_distributions.append(doc_distribution)
   topic_distributions = np.asarray(topic_distributions)
   return topic_distributions

def createMatrix(settings, model, bow_corpus, ids):

    df = pd.DataFrame(list(zip(ids, bow_corpus)), columns =['Date', 'val'])
    df = df.set_index(['Date'])
    df.sort_index(inplace=True)
    #Filter to relevant dates
    if 'start_date' in settings:
        df = df.loc[settings['start_date']:]
    if 'end_date' in settings:
        df = df.loc[:settings['end_date']]
    bow_corpus = df['val'].tolist()
    ids = df.index.values

    #Topics
    topic_distribution = model.show_topics(num_topics=settings['numberTopics'], num_words=settings['numberWords'],formatted=False)
    topics = []
    for topic_set in topic_distribution:
        topics.append(topic_set[0])

    # Output Topic Distribution and Topic Words
    print("Generate topic distribution data")
    distribution = get_doc_topic_distributions(model, bow_corpus)
    print("Get dominant topics")
    dominant_topic_counts, dominant_topic_dist = get_dominant_topics_counts_and_distribution(distribution)
    # print_dominant_topics_by_frequency(dominant_topic_counts, dominant_topic_dist)
    df1 = pd.DataFrame(distribution, index=ids, columns=topics)
    topic_data = []
    topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in topic_distribution]
    for topic, words_list in topics_words:
        topic_data.append(words_list + [dominant_topic_counts[topic]] + [np.around(dominant_topic_dist[topic], 4)])
    headers = [f"Word {i}" for i in range(0,len(words_list))] + ["Topic Count", "Distribution"]
    df2 = pd.DataFrame(topic_data, columns=headers)

    # Sort words by descending topic count
    df2 = df2.sort_values("Topic Count", ascending=False)
    save_df(settings['datasetName'], DISTRIB_FILE, df1)
    save_df(settings['datasetName'], WORDS_FILE, df2)

    return df1, df2
