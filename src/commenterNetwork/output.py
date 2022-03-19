import networkx as nx

from ..dataManager import checkPathExists, getFilePath


def saveNetwork(settings, file, G):
    file_path = getFilePath(settings, file)
    checkPathExists(file_path)
    nx.write_gexf(G, file_path)
