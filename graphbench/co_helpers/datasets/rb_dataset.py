import itertools
import random
import json
from typing import Optional, Union

import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

from .synthetic_dataset import SyntheticDataset



class RBDataset(SyntheticDataset):
    """
    A dataset of RB graphs.

    See
    Xu et al., "A simple model to generate hard satisfiable instances", 2005
    https://www.ijcai.org/Proceedings/05/Papers/0989.pdf

    Dataset parameters:
    - `num_cliques`: The number of cliques (integer, n >= 1, or tuple of two integers defining a range)
    - `k`: The number of nodes in each clique (integer, k >= 2, or tuple of two integers defining a range)
    - `p`: The tightness of each constraint.
      Regulates the interconnectedness between cliques, with a lower value corresponding to more connections between
      cliques
      (float, 1 >= p >= 0, or tuple of two floats defining a range)

    Optional parameters:
    - `num_nodes`: The number of nodes in the graph (tuple of two integers defining a range).
      Graphs will be re-sampled until the number of nodes falls into this range.
      Warning: Choosing this range carelessly can cause graph generation to get stuck in an endless loop of re-sampling.
    - `alpha`: Determines the domain size `d = n^alpha` of each variable (alpha > 0).
      Default: `log(k) / log(num_cliques)`
    - `r`: Determines the number `m = r * n * ln(n)` of constraints (r > 0). Default: `- alpha / log(1 - p)`
    """

    def __init__(
        self,
        root: str,
        num_samples: Optional[int] = None,
        num_cliques: Union[int, tuple[int, int]] = (20, 25),
        k: Union[int, tuple[int, int]] = (5, 12),
        p: Union[float, tuple[float, float]] = (0.3, 1),
        num_nodes: Optional[tuple[int, int]] = (200, 300),
        alpha: Optional[float] = None,
        r: Optional[float] = None,
        **kwargs
    ):
        self.num_cliques = num_cliques
        self.k = k
        self.p = p
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.r = r

        super().__init__(root, num_samples=num_samples, **kwargs)

    def create_graph(self, _index) -> tuple[Data, nx.Graph]:
        while True:
            num_cliques, k, r, p = self._prepare_parameters()

            sat_instance = generate_instance(num_cliques, k, r, p)
            graph_nx = nx.Graph()
            graph_nx.add_edges_from(sat_instance.clauses['NAND'])
            graph_nx.remove_nodes_from(list(nx.isolates(graph_nx)))

            if self.num_nodes is None \
                or self.num_nodes[0] <= graph_nx.number_of_nodes() <= self.num_nodes[1]:
                break

        graph_pyg = from_networkx(graph_nx)
        return graph_pyg, graph_nx

    def _prepare_parameters(self) -> tuple[int, int, float, float]:
        """
        Prepares the parameters `num_cliques`, `k`, `r`, and `p` necessary for sampling RB graphs.
        """
        # num_cliques
        if isinstance(self.num_cliques, int):
            num_cliques = self.num_cliques
        else:
            num_cliques = np.random.randint(self.num_cliques[0], self.num_cliques[1] + 1)

        # k
        if isinstance(self.k, int):
            k = self.k
        else:
            k = np.random.randint(self.k[0], self.k[1] + 1)

        # p
        if isinstance(self.p, float) or isinstance(self.p, int):
            p = self.p
        else:
            p = np.random.uniform(self.p[0], self.p[1])

        # alpha
        if self.alpha is not None:
            alpha = self.alpha
        else:
            alpha = np.log(k) / np.log(num_cliques)

        # r
        if self.r is not None:
            r = self.r
        else:
            r = - alpha / np.log(1 - p)

        return num_cliques, k, r, p



# Code below this point is copied from
# https://github.com/WenkelF/copt/blob/f61ed0376bd3b74e15b1ddd2afd4bd5e78570e35/graphgym/loader/dataset/rb_dataset.py

def generate_instance(n, k, r, p):
    v = k * n
    a = np.log(k) / np.log(n)
    s = int(p * (n ** (2 * a)))
    iterations = int(r * n * np.log(n) - 1)

    parts = np.reshape(np.int64(range(v)), (n, k))
    nand_clauses = []

    for i in parts:
        nand_clauses += itertools.combinations(i, 2)

    edges = set()
    for _ in range(iterations):
        i, j = np.random.choice(n, 2, replace=False)
        all = set(itertools.product(parts[i, :], parts[j, :]))
        all -= edges
        edges |= set(random.sample(tuple(all), k=min(s, len(all))))

    nand_clauses += list(edges)
    clauses = {'NAND': nand_clauses}

    instance = CSP_Instance(language=is_language,n_variables=v, clauses=clauses)
    return instance


class Constraint_Language:
    """ Class to represent a fixed Constraint Language """

    def __init__(self, domain_size, relations):
        """
        :param domain_size: Size of the underlying domain
        :param relations: A dict specifying the relations of the language. This also specifies a name for each relation.
                          I.E {'XOR': [[0, 1], [1, 0]], 'AND': [[1,1]]}
        """
        self.domain_size = domain_size
        self.relations = relations
        self.relation_names = list(relations.keys())

        # compute characteristic matrices for each relation
        self.relation_matrices = dict()
        for n, r in self.relations.items():
            M = np.zeros((self.domain_size, self.domain_size), dtype=np.float32)
            idx = np.array(r)
            M[idx[:, 0], idx[:, 1]] = 1.0
            self.relation_matrices[n] = M

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'domain_size': self.domain_size, 'relations': self.relations}, f, indent=4)

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = json.load(f)

        language = Constraint_Language(data['domain_size'], data['relations'])
        return language

    @staticmethod
    def get_coloring_language(d):

        def get_NEQ_relation(d):
            clauses = []
            for i in range(d):
                for j in range(d):
                    if not i == j:
                        clauses.append([i, j])
            return clauses

        lang = Constraint_Language(domain_size=d,
                                   relations={'NEQ': get_NEQ_relation(d)})
        return lang


# define constant constraint languages for Vertex Coloring, Independent Set and Max2Sat
coloring_language = Constraint_Language(domain_size=3,
                                        relations={'NEQ': [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]})

is_language = Constraint_Language(domain_size=2,
                                  relations={'NAND': [[0, 0], [0, 1], [1, 0]]})

max_2sat_language = Constraint_Language(domain_size=2,
                                        relations={'OR': [[0, 1], [1, 0], [1, 1]],
                                                   'IMPL': [[0, 0], [0, 1], [1, 1]],
                                                   'NAND': [[0, 0], [0, 1], [1, 0]]})

mc_weighted_language = Constraint_Language(domain_size=2,
                                           relations={'EQ': [[1, 1], [0, 0]], 'NEQ': [[1, 0], [0, 1]]})


class CSP_Instance:
    """ A class to represent a CSP instance """

    def __init__(self, language, n_variables, clauses, clause_weights=None, name=None):
        """
        :param language: A Constraint_Language object
        :param n_variables: The number of variables
        :param clauses: A dict specifying the clauses for each relation in the language.
                        I.E {'XOR': [[1,2], [5,4], [3,1]], 'AND': [[1,4], [2,5]]}
        """
        self.language = language
        self.n_variables = n_variables
        # assure clauses are un numpy format
        self.clauses = {r: np.int32(c) for r, c in clauses.items()}
        self.name = name

        if clause_weights is not None:
            self.weighted = True
            self.clause_weights = {r: np.float32(w) for r, w in clause_weights.items()}
        else:
            self.weighted = False

        # compute number of clauses and degree of each variable
        all_clauses = list(itertools.chain.from_iterable(clauses.values()))
        variables, counts = np.unique(all_clauses, return_counts=True)
        degrees = np.zeros(shape=(n_variables), dtype=np.int32)
        for u, c in zip(variables, counts):
            degrees[u] = c

        self.degrees = degrees
        self.n_clauses = len(all_clauses)

    def count_conflicts(self, assignment):
        """
        :param assignment: A hard variable assignment represented as a list of ints of length n_variables.
        :return: The number of unsatisfied clauses in this instances
        """
        conflicts = 0
        matrices = self.language.relation_matrices
        for r, M in matrices.items():
            valid = np.float32([M[assignment[u], assignment[v]] for [u, v] in self.clauses[r]])
            has_conflict = 1.0 - valid
            if self.weighted:
                has_conflict = has_conflict * self.clause_weights[r]

            conflicts += np.sum(has_conflict)

        return int(conflicts)

    @staticmethod
    def merge(instances):
        """
        A static function that merges multiple CSP instances into one
        :param instances: A list of CSP instances
        :return: CSP instances that contains all given instances with shifted variables
        """
        language = instances[0].language

        clauses = {r: [] for r in language.relation_names}
        n_variables = 0

        for instance in instances:
            for r in language.relation_names:
                shifted = instance.clauses[r] + n_variables
                clauses[r].append(shifted)
            n_variables += instance.n_variables

        clauses = {r: np.vstack(c) for r, c in clauses.items()}

        if instances[0].weighted:
            weights = {r: np.hstack([x.clause_weights[r] for x in instances]) for r in language.relation_names}
        else:
            weights = None

        merged_instance = CSP_Instance(language, n_variables, clauses, weights)
        return merged_instance

    @staticmethod
    def batch_instances(instances, batch_size):
        """
        Static method to merge given instances into batches
        :param instances: A list of CSP instances
        :param batch_size: The batch size
        :return: A list of CSP instances that each consist of 'batch_size' many merged instances
        """
        n_instances = len(instances)
        n_batches = int(np.ceil(n_instances / batch_size))
        batches = []

        print('Combining instances in batches...')
        for i in tqdm(range(n_batches)):
            start = i * batch_size
            end = min(start + batch_size, n_instances)
            batch_instance = CSP_Instance.merge(instances[start:end])
            batches.append(batch_instance)

        return batches

    @staticmethod
    def generate_random(n_variables, n_clauses, language, weighted=False):
        """
        :param n_variables: Number of variables
        :param n_clauses: Number of clauses
        :param language: A Constraint Language
        :return: A random CSP Instance with the specified parameters. Clauses are sampled uniformly.
        """
        variables = list(range(n_variables))
        clauses = {r: [] for r in language.relation_names}
        relations = np.random.choice(language.relation_names, n_clauses)

        for i in range(n_clauses):
            clause = list(np.random.choice(variables, 2, replace=False))
            r = relations[i]
            clauses[r].append(clause)

        if weighted:
            clause_weights = {r: np.random.uniform(size=[len(clauses[r])]) for r in language.relation_names}
            # clause_weights = {r: np.ones([len(clauses[r])]) for r in language.relation_names}
        else:
            clause_weights = None

        instance = CSP_Instance(language, n_variables, clauses, clause_weights)
        return instance

    @staticmethod
    def graph_to_csp_instance(graph, language, relation_name, name=None):
        """
        :param graph: A NetworkX graphs
        :param language: A Constraint Language
        :param relation_name: The relation name to assign to each edge
        :return: A CSP Instance representing the graph
        """
        adj = nx.linalg.adjacency_matrix(graph)
        n_variables = adj.shape[0]
        clauses = {relation_name: np.int32(graph.edges())}

        instance = CSP_Instance(language, n_variables, clauses, name=name)
        return instance

    @staticmethod
    def graph_to_weighted_mc_instance(graph, name=None):
        """
        :param graph: A NetworkX graphs
        :param language: A Constraint Language
        :param relation_name: The relation name to assign to each edge
        :return: A CSP Instance representing the graph
        """
        adj = nx.linalg.adjacency_matrix(graph)
        n_variables = adj.shape[0]
        clauses = {'EQ': [], 'NEQ': []}
        for u, v, w in graph.edges(data='weight'):
            rel = 'NEQ' if w > 0 else 'EQ'
            clauses[rel].append([u, v])

        instance = CSP_Instance(mc_weighted_language, n_variables, clauses, name=name)
        return instance

    @staticmethod
    def cnf_to_instance(formula, clause_weights=None):
        """
        :param formula: A 2-cnf formula represented as a list of lists of ints.
                        I.e. ((X1 or X2) and (not X2 or X3)) is [[1, 2], [-2, 3]]
        :return: A CSP instance that represents the formula
        """

        def clause_type(clause):
            # returns the relation type for a given clause
            if clause[0] * clause[1] < 0:
                return 'IMPL'
            elif clause[0] > 0:
                return 'OR'
            else:
                return 'NAND'

        def normalize_2SAT_clauses(formula):
            # Transforms clauses of form [v, -u] to [-u, v]. This unifies the direction of all implication clauses.
            fill_monom_clause = lambda c: [c[0], c[0]] if len(c) == 1 else c
            filled_formula = list(map(fill_monom_clause, formula))
            normalize_impl_clause = lambda c: [c[1], c[0]] if clause_type(c) == 'IMPL' and c[0] > 0 else c
            normed_formula = list(map(normalize_impl_clause, filled_formula))
            return normed_formula

        formula = normalize_2SAT_clauses(formula)

        clauses = {t: [] for t in {'OR', 'IMPL', 'NAND'}}

        weighted = clause_weights is not None
        if weighted:
            weights = {t: [] for t in {'OR', 'IMPL', 'NAND'}}
        else:
            weights = None

        for i, c in enumerate(formula):
            u = abs(c[0]) - 1
            v = abs(c[1]) - 1
            t = clause_type(c)
            clauses[t].append([u, v])
            if weighted:
                weights[t].append(clause_weights[i])

        n_variables = np.max([np.max(np.abs(clause)) for clause in formula])

        instance = CSP_Instance(max_2sat_language, n_variables, clauses, clause_weights=weights)
        return instance
