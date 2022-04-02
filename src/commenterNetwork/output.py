import networkx as nx

from ..dataManager import checkPathExists, getFilePath


def saveNetwork(settings, file, G):
    """Save the networkx graph to disk."""
    file_path = getFilePath(settings, file)
    checkPathExists(file_path)
    nx.write_gexf(G, file_path)
