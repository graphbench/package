import random

import networkx as nx
import torch
import tqdm
from networkx.algorithms import tree
from torch_geometric.utils import from_networkx


# -----------------------------------------------------------------------------
# Static Variables for Graph Generation and Algorithm Configuration
# -----------------------------------------------------------------------------

"""
GENERATORS (dict):
    Maps string names of graph generator types to their corresponding NetworkX generator functions.
    Used to dynamically select and instantiate different random graph models.
"""
GENERATORS = {
    "erdos-reyni": nx.erdos_renyi_graph,
    "newman–watts–strogatz": nx.generators.random_graphs.newman_watts_strogatz_graph,
    "barabasi-albert": nx.generators.random_graphs.barabasi_albert_graph,
    "dual-barabasi-albert": nx.generators.random_graphs.dual_barabasi_albert_graph,
    "powerlaw-cluster": nx.generators.random_graphs.powerlaw_cluster_graph,
    "stochastic-block-model": nx.stochastic_block_model,
}

"""
GENERATORS_LIST (list):
    List of all available graph generator names as strings.
    Useful for sampling or iterating over supported generator types.
"""
GENERATORS_LIST = list(GENERATORS.keys())

"""
CONFIG (dict):
    Configuration dictionary for each graph generator and algorithm type.
    Each entry specifies the parameters used for training data generation for different algorithms.
    Format and parameters depend on the generator type (see inline comments for details).
"""
CONFIG = {
    "erdos-reyni": {"bridges": (1,0.11), "shortest_path": (1,0.17), "mst": (1,0.19), "flow": (1,0.16), "maxclique":(1,0.9), "steinertree":(1,0.14), "bipartitematching":(1,0.08), "topologicalorder":(1,0.3)},
    "newman–watts–strogatz": {"bridges": (1,1,1), "shortest_path": (1,2,0.15), "mst": (1,4,0.2), "flow": (1,4,0.2), "maxclique":(1,4,0.6), "steinertree":(1,4,0.12), "bipartitematching":(1,6,0.18), "topologicalorder":(1,2,0.2)},
    "barabasi-albert": {"bridges": (1,1), "shortest_path": (1,2), "mst": (1,3), "flow": (1,3), "maxclique":(1,8), "steinertree":(1,2), "bipartitematching":(1,3), "topologicalorder":(1,4)},
    "dual-barabasi-albert": {"bridges": (1,3,1,0.07), "shortest_path": (1,2,1,0.05), "mst": (1,4,2,0.3), "flow": (1,4,2,0.3), "maxclique":(1,4,2,0.3), "steinertree":(1,3,1,0.4), "bipartitematching":(1,4,2,0.2), "topologicalorder":(1,3,2,0.5)},
    "powerlaw-cluster": {"bridges": (1,1,0.5), "shortest_path": (1,1,0.35), "mst": (1,5,0.4), "flow": (1,5,0.4), "maxclique":(1,9,0.5), "steinertree":(1,3,0.7), "bipartitematching":(1,8,0.5), "topologicalorder":(1,3,0.1)},
    "stochastic-block-model": {"bridges": (1,[0.5,0.5], [[0.5,0.01],[0.01,0.5]]), "shortest_path": (1,[0.5,0.5], [[0.31,0.01],[0.01,0.31]]), "mst": (1,[0.5,0.5],[[0.5, 0.3],[0.3,0.5]]), "flow": (1,[0.5,0.5],[[0.35, 0.3],[0.35,0.3]]), "maxclique":(1,[0.5,0.5],[[0.75,0.75],[0.75,0.75]]), "steinertree":(1,[0.5,0.5],[[0.4,0.4],[0.4,0.4]]), "bipartitematching":(1,[0.5,0.5],[[0.31,0.1],[0.1,0.31]]), "topologicalorder":(1, [0.5,0.5],[[0.2,0.2],[0.2,0.2]])}
}

"""
CONFIG_TEST (dict):
    Configuration dictionary for each graph generator and algorithm type for test data.
    Each entry specifies the parameters used for test data generation for different algorithms.
    Format and parameters depend on the generator type (see inline comments for details).
"""
CONFIG_TEST = {
    "erdos-reyni": {"bridges": (1,0.07), "shortest_path": (1,0.17), "mst": (1,0.25), "flow": (1,0.16), "maxclique":(1,0.9), "steinertree":(1,0.14), "bipartitematching":(1,0.08), "topologicalorder":(1,0.3)},
    "newman–watts–strogatz": {"bridges": (1,1,1), "shortest_path": (1,2,0.15), "mst": (1,5,0.8), "flow": (1,4,0.2), "maxclique":(1,4,0.6), "steinertree":(1,4,0.12), "bipartitematching":(1,6,0.18), "topologicalorder":(1,2,0.2)},
    "barabasi-albert": {"bridges": (1,1), "shortest_path": (1,2), "mst": (1,7), "flow":(1,3), "maxclique":(1,8), "steinertree":(1,2), "bipartitematching":(1,3), "topologicalorder":(1,4)},
    "dual-barabasi-albert": {"bridges": (1,4,1,0.6), "shortest_path": (1,2,1,0.05), "mst": (1,4,3,0.6), "flow": (1,4,2,0.3), "maxclique":(1,4,2,0.3), "steinertree":(1,3,1,0.4), "bipartitematching":(1,4,2,0.2), "topologicalorder":(1,3,2,0.5)},
    "powerlaw-cluster": {"bridges": (1,1,0.8), "shortest_path": (1,1,0.35), "mst": (1,7,0.7), "flow": (1,5,0.4), "maxclique":(1,9,0.5), "steinertree":(1,3,0.7), "bipartitematching":(1,8,0.5), "topologicalorder":(1,3,0.1)},
    "stochastic-block-model": {"bridges": (1,[0.5,0.5], [[0.05,0.001],[0.001,0.05]]), "shortest_path": (1,[0.5,0.5], [[0.31,0.01],[0.01,0.31]]), "mst": (1,[0.4,0.6],[[0.25, 0.65],[0.65,0.25]]), "flow": (1,[0.5,0.5],[[0.35, 0.3],[0.35,0.3]]), "maxclique":(1,[0.5,0.5],[[0.75,0.75],[0.75,0.75]]), "steinertree":(1,[0.5,0.5],[[0.4,0.4],[0.4,0.4]]), "bipartitematching":(1,[0.5,0.5],[[0.31,0.1],[0.1,0.31]]), "topologicalorder":(1, [0.5,0.5],[[0.2,0.5],[0.5,0.2]])}
}

"""
SAMPLES (dict):
    Specifies the number of samples to generate for each dataset split ('train', 'val', 'test').
"""
SAMPLES = {
    "train": 1000000,
    "val": 10000,
    "test": 10000,
}

"""
SAMPLING_LIST_TRAIN (dict):
    Sampling weights for each difficulty level during training.
    Each list corresponds to the probability weights for selecting each generator type.
"""
SAMPLING_LIST_TRAIN = {
    "easy": [1,0,1,0,1,0],
    "medium": [1,0,0,0,0,0],
    "hard": [1,0,0,0,0,0],
}

"""
SAMPLING_LIST_TEST (dict):
    Sampling weights for each difficulty level during testing.
    Each list corresponds to the probability weights for selecting each generator type.
"""
SAMPLING_LIST_TEST = {
    "easy": [1,1,1,1,1,1],
    "medium": [1,1,1,1,1,1],
    "hard": [0,1,1,1,1,1],
}

def generate_graph_util(num_nodes, name, generator, is_training):
    if generator == "stochastic-block-model":
        if is_training:
            graph = GENERATORS[generator]([int(CONFIG[generator][name][1][0]*num_nodes), int(CONFIG[generator][name][1][1]*num_nodes)], CONFIG[generator][name][2])
        else:
            graph = GENERATORS[generator]([int(CONFIG_TEST[generator][name][1][0]*num_nodes), int(CONFIG_TEST[generator][name][1][1]*num_nodes)], CONFIG_TEST[generator][name][2])
    else:
        if is_training:
            graph = GENERATORS[generator](num_nodes, *CONFIG[generator][name][1:])
        else:
            graph = GENERATORS[generator](num_nodes, *CONFIG_TEST[generator][name][1:])
    cc = list(nx.connected_components(graph))
    if len(cc) >= 2:
        for i in range(len(cc)):
            for _ in range(CONFIG[generator][name][0]):
                j = random.choice([j for j in range(len(cc)) if j != i])
                a = random.choice(list(cc[i]))
                b = random.choice(list(cc[j]))
                graph.add_edge(a, b)
    return graph


def generate_graph(sampling_list, num_nodes, name, is_training):
    random_generator = random.choices(GENERATORS_LIST, sampling_list, k=2)[0]

    return generate_graph_util(num_nodes=num_nodes, name=name, generator=random_generator, is_training=is_training)




def mst_graph(num_nodes, name, sampling_list, is_training):
    while True:
        graph = generate_graph(sampling_list, num_nodes, name, is_training)
        if len(graph.edges) < 1:
            continue

        weight_dict = {
            e: {"edge_attr": round(random.uniform(0, 10), 6)} for e in graph.edges
        }

        weights = [v["edge_attr"] for k, v in weight_dict.items()]
        if len(weights) == len(set(weights)):
            nx.set_edge_attributes(graph, weight_dict)
            edges = [
                (u, v)
                for u, v, _ in tree.minimum_spanning_edges(graph, weight="edge_attr")
            ]
            return graph, edges


def mst(num_nodes, name, sampling_list, is_training):
    graph, edges = mst_graph(num_nodes, name, sampling_list, is_training)
    data = from_networkx(graph)
    data.y = torch.zeros(data.edge_index.size(1), dtype=torch.long)

    for i in range(data.edge_index.size(1)):
        if (data.edge_index[0, i], data.edge_index[1, i]) in edges:
            data.y[i] = 1
    data.edge_attr = data.edge_attr[:, None]
    return data



def bridges_graph(num_nodes, name, sampling_list, is_training):
    graph = generate_graph(sampling_list, num_nodes, name, is_training)
    edges = list(nx.bridges(graph))
    return graph, edges


def bridges(num_nodes, name, sampling_list, is_training):
    graph, edges = bridges_graph(num_nodes, name, sampling_list, is_training)
    data = from_networkx(graph)
    data.y = torch.zeros(data.edge_index.size(1), dtype=torch.long)

    for i in range(data.edge_index.size(1)):
        if (data.edge_index[0, i], data.edge_index[1, i]) in edges:
            data.y[i] = 1

    return data



def flow_graph(sampling_list, num_nodes, name, is_training):
    graph = generate_graph(sampling_list, num_nodes, name, is_training).to_directed()
    weight_dict = {
        e: {"edge_attr": round(random.uniform(0, 3), 2)} for e in graph.edges
    }
    nx.set_edge_attributes(graph, weight_dict)
    nodes = list(range(num_nodes))
    source = random.choice(nodes)
    nodes = [n for n in nodes if n != source]
    sink = random.choice(nodes)
    value = nx.flow.maximum_flow_value(graph, source, sink, capacity="edge_attr")
    return graph, source, sink, value


def flow(num_nodes, name, sampling_list, is_training):
    graph, source, sink, value = flow_graph(sampling_list, num_nodes, name, is_training)
    data = from_networkx(graph)
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    data.y = torch.tensor(value)
    data.x[source] = 1
    data.x[sink] = 2
    data.edge_attr = data.edge_attr[:, None]
    return data



def max_clique_graph(num_nodes, name, sampling_list, is_training):
    graph = generate_graph(sampling_list, num_nodes, name, is_training)

    max_clique = nx.approximation.max_clique(graph)
    
    return graph, max_clique


def max_clique(num_nodes, name, sampling_list, is_training):
    graph, max_clique_nodes = max_clique_graph(num_nodes, name, sampling_list, is_training)
    
    data = from_networkx(graph)
    
    data.y = torch.zeros(data.num_nodes, dtype=torch.long)
    
    max_clique_indices = torch.tensor(list(max_clique_nodes), dtype=torch.long)
    if len(max_clique_indices) > 0:
        data.y[max_clique_indices] = 1
    
    return data



def steiner_tree_graph(num_nodes, name, sampling_list, is_training, num_terminals=3):
    while True:
        graph = generate_graph(sampling_list, num_nodes, name, is_training)
        if len(graph.edges) < 1:
            continue

        weight_dict = {
            e: {"edge_attr": round(random.uniform(0, 10), 6)} for e in graph.edges
        }
        
        weights = [v["edge_attr"] for k, v in weight_dict.items()]
        if len(weights) == len(set(weights)):
            nx.set_edge_attributes(graph, weight_dict)
            
            terminal_nodes = random.sample(list(graph.nodes), min(num_terminals, len(graph.nodes)))
            
            try:
                steiner_tree = nx.algorithms.approximation.steiner_tree(graph, terminal_nodes, weight="edge_attr")
                steiner_edges = list(steiner_tree.edges)
                return graph, steiner_edges, terminal_nodes
            except:
                continue


def steiner_tree(num_nodes, name, sampling_list, is_training, num_terminals=3):
    graph, steiner_edges, terminal_nodes = steiner_tree_graph(num_nodes, name, sampling_list, is_training, num_terminals)
    
    data = from_networkx(graph)
    
    data.y = torch.zeros(data.edge_index.size(1), dtype=torch.long)
    
    for i in range(data.edge_index.size(1)):
        edge = (data.edge_index[0, i].item(), data.edge_index[1, i].item())
        if edge in steiner_edges or (edge[1], edge[0]) in steiner_edges:
            data.y[i] = 1
    
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    terminal_indices = torch.tensor(terminal_nodes, dtype=torch.long)
    data.x[terminal_indices] = 1  
    
    data.edge_attr = data.edge_attr[:, None]
    
    return data



def bipartite_matching_graph(num_nodes, name, sampling_list, is_training, p=0.3):

    graph = generate_graph(sampling_list, num_nodes, name, is_training)

    weight_dict = {
        e: {"edge_attr": round(random.uniform(0, 5), 2)} for e in graph.edges
    }
    nx.set_edge_attributes(graph, weight_dict)
    
    matching = nx.max_weight_matching(graph, weight="edge_attr")
    matching_edges = list(matching)
        
    return graph, matching_edges
    


def bipartite_matching(num_nodes, name=None, sampling_list=None, is_training=None, p=0.3, *args, **kwargs):
    graph, matching_edges = bipartite_matching_graph(num_nodes, name, sampling_list, is_training)
    
    data = from_networkx(graph)
    
    data.y = torch.zeros(data.edge_index.size(1), dtype=torch.long)
    
    for i in range(data.edge_index.size(1)):
        edge = (data.edge_index[0, i].item(), data.edge_index[1, i].item())
        if edge in matching_edges or (edge[1], edge[0]) in matching_edges:
            data.y[i] = 1
    
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    data.edge_attr = data.edge_attr[:, None]
    return data


def topological_order_graph(num_nodes, name, sampling_list, is_training):
    undirected = generate_graph(sampling_list, num_nodes, name, is_training)
    order = list(range(num_nodes))
    random.shuffle(order)
    position = {node: idx for idx, node in enumerate(order)}
    dag = nx.DiGraph()
    dag.add_nodes_from(undirected.nodes)
    for u, v in undirected.edges:
        if position[u] < position[v]:
            dag.add_edge(u, v)
        elif position[v] < position[u]:
            dag.add_edge(v, u)

    topo = list(nx.topological_sort(dag))
    rank = {node: idx for idx, node in enumerate(topo)}
    return dag, rank


def topological_order(num_nodes, name, sampling_list, is_training):
    graph, rank = topological_order_graph(num_nodes, name, sampling_list, is_training)
    data = from_networkx(graph)
    y = torch.zeros(data.num_nodes, dtype=torch.float)
    if data.num_nodes > 1:
        for node, r in rank.items():
            y[node] = r / (data.num_nodes - 1)
    data.y = y
    return data





ALGORITHMS = {
    "bridges": bridges,
    "mst": mst,
    "flow": flow,
    "maxclique": max_clique,
    "steinertree": steiner_tree,
    "bipartitematching": bipartite_matching,
    "topologicalorder": topological_order,
}


def generate_algoreas_data(name, num_nodes, difficulty, split):
    
    if split == "train":
        data_list = [
                ALGORITHMS[name](num_nodes, name, SAMPLING_LIST_TRAIN[difficulty], True)
                for _ in tqdm.tqdm(range(SAMPLES[split]))
            
            ]
    else: 
        data_list = [
                ALGORITHMS[name](num_nodes, name, SAMPLING_LIST_TEST[difficulty], False)
                for _ in tqdm.tqdm(range(SAMPLES[split]))
            
            ]

    return data_list