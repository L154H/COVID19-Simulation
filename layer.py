import networkx as nx
import numpy as np
import random
from typing import List
from collections import Counter
from copy import deepcopy
from util import timing


###############################################################################
#     LayerGraph                                                              #
###############################################################################
class LayerGraph(nx.Graph):
    """
    Layergraph where we can add and remove layers and the graph gets the
    corresponging edges
    """
    def __init__(self, num_nodes: int, categories: List[str],
                 percentage: List[float] = None, category_of=None):
        nx.Graph.__init__(self)
        self.num_nodes = num_nodes
        self.add_nodes_from(range(num_nodes))
        self.categories = categories
        self.layers = {}            # maps from layername to layer
        self.countedges = {}        # count appearances of edge in all layers

        if category_of is None:
            if percentage is None:
                # all categories equally distributed
                percentage = [1/(len(categories))]*len(categories)
            distribution = np.random.choice(
                categories, size=num_nodes, p=percentage
            )
            self.category_of = {
                node: distribution[node] for node in self.nodes()
            }
        else:
            self.category_of = category_of
        # dictionary that maps from category to list of nodes with category
        self.nodes_in_category = {category: [] for category in categories}
        for node in self.nodes():
            category = self.category_of[node]
            self.nodes_in_category[category].append(node)

    def get_n_categories_category_of(self):
        return (self.num_nodes, self.categories, self.category_of)

    def add_layer(self, layer, p=1):
        self.layers[layer.name] = layer
        self.update_layer(layer.name, p)

    def update_layer(self, layername, percentage):
        """
        activates just a percentage of edges in layer
        """
        layer = self.layers[layername]
        self.remove_all_edges_from_layer(layer)
        layer.set_active(percentage)
        for edge in layer.active_edges:
            self.add_edge_and_count(edge)

    def remove_layer(self, layer):
        self.update_layer(layer.name, 0)
        del self.layers[layer.name]

    def remove_all_edges_from_layer(self, layer):
        for edge in layer.active_edges:
            self.remove_edge_and_count(edge)

    def add_edge_and_count(self, edge):
        edge = order(edge)
        if edge in self.countedges:
            self.countedges[edge] += 1
        else:
            self.countedges[edge] = 1
            self.add_edge(*edge)

    def remove_edge_and_count(self, edge):
        edge = order(edge)
        self.countedges[edge] -= 1
        if self.countedges[edge] <= 0:
            self.remove_edge(*edge)
            del self.countedges[edge]


###############################################################################
#     LayerFactory                                                            #
###############################################################################
class LayerFactory():
    """
    Factory that builds layers for a layer graph
    """
    def __init__(self, LG):
        self.LG = LG

    @timing
    def layer_dividing_Graph(self, name, group_size_mean, degree, categories,
                             fully_connected=False):
        union = []
        for category in categories:
            union.extend(self.LG.nodes_in_category[category])
        random.shuffle(union)
        index = 0
        groups = []
        while index < len(union):
            group_size = np.random.poisson(lam=group_size_mean - 1) + 1
            group = union[index: index + group_size]
            index = index + group_size
            groups.append(group)

        edges = self.get_edges(groups, degree, fully_connected=fully_connected)

        return Layer(name, groups, edges)

    @timing
    def create_layer(self, name: str, group_size_mean: int, num_groups: int,
                     percentage_of_category: List[float], degree,
                     fully_connected=False, disjoint=True):
        """
        Returns a layer with the given properties

        Arguments:
            name: name of the layer
            group_size_mean: mean of the group sizes of the layers
            num_groups: number of groups in layer
            percentage_of_category: fraction of each category in one group
            Ex. : [0.4, 0.3, 0.3] while categories = ["kids", "normal", "risk"]
            degree: average degree of nodes in each group
        Keyword arguments:
            fully_connected: groups in layer fully connected
            disjoint: groups are disjoint
        """
        msg = (
            "Length of category and percentage are not the same!"
            + str(len(self.LG.categories))
            + str(len(percentage_of_category))
        )
        assert len(percentage_of_category) == len(self.LG.categories), msg
        groups = [[] for _ in range(num_groups)]
        unused_nodes = deepcopy(self.LG.nodes_in_category)

        for index_group in range(num_groups):
            group_size = np.random.poisson(lam=group_size_mean - 1) + 1
            group_cat = np.random.choice(
                self.LG.categories, size=group_size, p=percentage_of_category
            )
            size_of_category = Counter(group_cat)
            for cat in self.LG.categories:
                if len(unused_nodes[cat]) < size_of_category[cat]:
                    unused_nodes[cat] = self.LG.nodes_in_category[cat].copy()
                if size_of_category[cat] >= len(unused_nodes[cat]):
                    nodes = unused_nodes[cat]
                else:
                    nodes = np.random.choice(
                        unused_nodes[cat], size=size_of_category[cat],
                        replace=False
                    )
                for node in nodes:
                    groups[index_group].append(node)
                    if disjoint:
                        unused_nodes[cat].remove(node)
        edges = self.get_edges(groups, degree, fully_connected)
        return Layer(name, groups, edges)

    def get_edges(self, groups, degree, fully_connected=False):
        """
        Returns list of edges of groups for layer

        Arguments:
            groups: groups of layer
            degree: average degree of nodes in each group
        Keyword arguments:
            fully_connected: groups in layer fully connected
        """
        edges = []
        for nodesgroup in groups:
            n = len(nodesgroup)
            if n == 1:
                continue
            if fully_connected:
                H = nx.complete_graph(n)
            else:
                p = degree / (n-1)
                H = nx.fast_gnp_random_graph(n, p)
            for i1, i2 in H.edges():
                edge = order((nodesgroup[i1], nodesgroup[i2]))
                edges.append(edge)
        return edges


###############################################################################
#     Layer                                                                   #
###############################################################################
class Layer():
    """
    One layer for the graph construction
    """
    def __init__(self, name, groups, edges):
        self.name = name
        self.groups = groups
        self.all_edges = edges
        self.active_edges = []

    def copy_Layer(self):
        return Layer(self.name, self.groups, self.all_edges)

    def set_active(self, p):
        num_all_edges = int(len(self.all_edges))
        num_active_edges = int(round(num_all_edges * p))
        active_edges_indices = np.random.choice(
            num_all_edges, size=num_active_edges, replace=False
        )
        self.active_edges = [
            list(self.all_edges)[index] for index in active_edges_indices
        ]


###############################################################################
#     Usefull Functions                                                       #
###############################################################################

def order(edge):
    a, b = edge
    if a < b:
        edge = (a, b)
    else:
        edge = (b, a)
    return edge
