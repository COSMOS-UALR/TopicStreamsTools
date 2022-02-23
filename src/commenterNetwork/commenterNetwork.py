from collections import Counter
from itertools import combinations
from functools import partial
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import networkx.algorithms.community as nx_comm
import os
from tqdm import tqdm
from p_tqdm import p_map

from ..dataManager import fileExists

# import cufflinks as cf
# init_notebook_mode(connected=True)
# cf.go_offline()

# pd.set_option("display.max_rows", 999)
# pd.set_option("display.max_columns", 999)


def preprocess(file):
    DATA_FOLDER = "Data"
    path = os.path.join(DATA_FOLDER, file)
    df = pd.read_csv(path)
    df1 = df[["commenter_id", "video_id"]]
    df = df1.groupby(['commenter_id', 'video_id']).size().reset_index(name='counts')
    df.columns = ['source', 'target', 'strength']
    df = df[['source', 'target']]
    total_target = df.target.nunique()
    return df


def toEdgeList(df):
    """From comment data to edge-lists"""
    weight_threshold = 1
    # weight_threshold = 10
    chunk_size = 100000
    edges_list_weigths = []
    for start in tqdm(range(0, df.shape[0], chunk_size)):
        df_subset = df.iloc[start : start + chunk_size]
        gp = df_subset.groupby('target', as_index=False).aggregate(lambda tdf: tdf.unique().tolist())
        parallel_list = gp.apply(lambda s: [] if len(s.iloc[1]) <= 1 else list(combinations(sorted(s.iloc[1]), 2)), axis=1)
        edges = [element for list_val in parallel_list.to_list() for element in list_val]
        if len(edges) > 0:
            ht_pair_count = Counter(edges)
            edges_list_weigths.extend([(ht1, ht2, w) for (ht1, ht2), w in ht_pair_count.items() if w >= weight_threshold])
    df_comm = pd.DataFrame(data=edges_list_weigths, columns=['Source', 'Target', 'Weight'])
    df_comm = df_comm.groupby(['Source', 'Target'])['Weight'].sum().reset_index()
    df_comm = df_comm[df_comm['Weight'] >= 10]
    df_comm.reset_index(drop=True, inplace=True)
    return df_comm


def edgeListToNetwork(df):
    """From edge-lists to networks"""
    if 'Unnamed: 0' in df.columns:
        df.drop(['Unnamed: 0'], axis=1)
    G = nx.Graph()
    G = nx.from_pandas_edgelist(df, source='Source', target='Target', create_using=nx.Graph())
    return G


def printGraphInfo(G):
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    # normalized = nodes / total_commenters
    modularity = nx_comm.modularity(G, nx_comm.greedy_modularity_communities(G))
    try:
        clus = nx.average_clustering(G)
    except:
        clus = "NaN"
    print('nodes: ', nodes, 'edges: ', edges, "clus: ", clus, 'modularity: ', modularity)
    try:
        print("average shortest p. l.: ", nx.average_shortest_path_length(G))
    except:
        pass
    try:
        degrees = dict(G.degree())
        Network_Avg_Degree = round((sum(list(degrees.values())) / len(G)), 3)
        print("average degree: ", Network_Avg_Degree)
    except:
        pass
    try:
        print("density: ", nx.density(G))
    except:
        pass


def maximal_cliques(G, n):
    """
    Networkx's find_cliques returns an iterator over maximal cliques, each of which is a list of nodes in G. The order of cliques is arbitrary.
    For each node v, a maximal clique for v is a largest complete subgraph containing v. The largest maximal clique is sometimes called the maximum clique.
    """
    mcs = []
    for clique in nx.find_cliques(G):
        if len(clique) == n:
            mcs.append(clique)
    return mcs


def count(i, G, size_list, count_list):
    count = len(maximal_cliques(G, i))
    if count > 0:
        size_list.append(i)
        count_list.append(count)
        return i, count

    
def findCliques(G):
    size_list = []
    count_list = []
    count_list_check = []
    f = partial(count, G=G, size_list=size_list, count_list=count_list)
    count_list_check = p_map(f, range(100))
    count_list = [i for i in count_list_check if i]
    return pd.DataFrame(count_list, columns =['Size', 'Count'])


def getDfCheck(i, G):
    if len(i) >= 5:
        deg_list = []
        for node in i:
            deg_list.append(G.degree[node])
        clus_list = list(nx.clustering(G, i).values())
        return len(i), i, round((sum(deg_list) / len(deg_list)), 3),round((sum(clus_list) / len(clus_list)), 3) 


def calculateCliqueStats(gl, G):
    f = partial(getDfCheck, G=G)
    df_check = p_map(f, gl)
    df_check = [i for i in df_check if i]
    df = pd.DataFrame(df_check, columns =['Clique_Size', 'Clique_Members', 'Average_Degree', 'Avg_Clus_Coeff'])
    df['Network_Avg_Clus_Coeff'] = round(nx.average_clustering(G), 3)
    degrees = dict(G.degree())
    df['Network_Avg_Degree'] = round((sum(list(degrees.values())) / len(G)), 3)
    df['Normalized_Avg_Clus_Coeff'] = df['Avg_Clus_Coeff'] / df['Network_Avg_Clus_Coeff']
    df['Normalized_Avg_Degree'] = df['Average_Degree'] / df['Network_Avg_Degree']
    df = df[['Clique_Size', 'Clique_Members', 'Average_Degree', 'Network_Avg_Degree', 'Normalized_Avg_Degree', 'Avg_Clus_Coeff', 'Network_Avg_Clus_Coeff', 'Normalized_Avg_Clus_Coeff']]
    return df


def drawFigures(df_groupby, cliques):
    fig = px.bar(df_groupby, x='Clique_Size', y='ratio', title='Clique Size Distribution')
    fig.show()
    fig = px.bar(cliques, x='Size', y='Count', title='Clique Size Distribution')
    fig.show()
    df_groupby['ratio'].sum()


def showCliqueStats(df):
    df.describe()
    len(df["Clique_Members"].explode().unique())
    df['Avg_Clus_Coeff'].mean()
    df['Normalized_Avg_Clus_Coeff'].mean()
    df['Average_Degree'].mean()
    df['Normalized_Avg_Degree'].mean()
    df.shape
    df.Clique_Size.mean()
    df.Clique_Size.max()
    df.Clique_Size.median()


def commenterNetwork():
    file = "UCaBf1a-dpIsw8OxqH4ki2Kg_Comments.csv"
    if fileExists('network.gexf'):
        G = nx.read_gexf('network.gexf')
    else:
        df = preprocess(file)
        # total_commenters = df.source.nunique()
        df_comm = toEdgeList(df)
        G = edgeListToNetwork(df_comm)
        nx.write_gexf(G, 'network.gexf')
    # printGraphInfo(G)
    if fileExists('cliques_count.csv'):
        df_cliques_count = pd.read_csv('cliques_count.csv')
    else:
        df_cliques_count = findCliques(G)
        df_cliques_count.to_csv('cliques_count.csv')
    gl = list(nx.find_cliques(G))
    if fileExists('cliques.csv'):
        df_cliques = pd.read_csv('cliques.csv')
    else:
        df_cliques = calculateCliqueStats(gl, G)
        df_cliques.to_csv('cliques.csv')
    df_groupby = df_cliques.groupby(['Clique_Size']).size().reset_index(name='counts')
    df_groupby['ratio'] = df_groupby.counts / df_cliques_count.Count.sum()
    df_groupby
    np.sum(df_groupby.counts)
    drawFigures(df_groupby, df_cliques_count)
    showCliqueStats(df_cliques)
