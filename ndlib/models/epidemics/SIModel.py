from ..DiffusionModel import DiffusionModel
import numpy as np
import networkx as nx
import future.utils

__author__ = "Giulio Rossetti"
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"


class SIModel(DiffusionModel):
    """
    Model Parameters to be specified via ModelConfig

    :param beta: The infection rate (float value in [0,1])
    """

    def __init__(self, graph, seed=None):
        """
        Model Constructor

        :param graph: A networkx graph object
        """
        super(self.__class__, self).__init__(graph, seed)
        self.available_statuses = {"Susceptible": 0, "Infected": 1}

        self.parameters = {
            "model": {
                "beta": {
                    "descr": "Infection rate",
                    "range": "[0,1]",
                    "optional": False,
                },
                "tp_rate": {
                    "descr": "Whether if the infection rate depends on the number of infected neighbors",
                    "range": [0, 1],
                    "optional": True,
                    "default": 1,
                },
            },
            "nodes": {},
            "edges": {},
        }

        self.name = "SI"

    def iteration(self, node_status=True):
        """
        Execute a single model iteration

        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        self.clean_initial_status(self.available_statuses.values())

        actual_status = {
            node: nstatus for node, nstatus in future.utils.iteritems(self.status)
        }

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(actual_status)
            if node_status:
                return {
                    "iteration": 0,
                    "status": actual_status.copy(),
                    "node_count": node_count.copy(),
                    "status_delta": status_delta.copy(),
                }
            else:
                return {
                    "iteration": 0,
                    "status": {},
                    "node_count": node_count.copy(),
                    "status_delta": status_delta.copy(),
                }
            

        has_weights = self.graph_has_weights
        beta = self.params["model"]["beta"]
        use_tp = self.params["model"]["tp_rate"] == 1
        actual_status = self.status.copy()  # Assuming `actual_status` starts as a copy of `self.status`.


        # Pre-fetch edge weights if needed
        all_weights = self.graph.get_edge_attributes("weight") if has_weights else None

        # Precompute neighbors for all nodes
        if self.graph.directed:
            neighbors_dict = {u: list(self.graph.predecessors(u)) for u in self.graph.nodes}
        else:
            neighbors_dict = {u: list(self.graph.neighbors(u)) for u in self.graph.nodes}

        # Precompute random numbers for all nodes
        random_numbers = np.random.random_sample(len(self.graph.nodes))

        for idx, u in enumerate(self.graph.nodes):
            if self.status[u] == 0:  # Only process susceptible nodes
                eventp = random_numbers[idx]
                
                # Get neighbors depending on graph type
                neighbors = neighbors_dict[u]
                
                # Find infected neighbors
                infected_neighbors = [
                    v for v in neighbors if self.status[v] == 1
                ]
                if not infected_neighbors:
                    continue  # Skip if no infected neighbors
                
                # Compute infection probability
                if use_tp:
                    if has_weights:
                        # Weighted sum of neighbor infections
                        # Accumulate weights once
                        total_weight = sum(all_weights.get((v,u), 1.0) 
                                        for v in infected_neighbors)
                        infection_prob = 1 - (1 - beta) ** total_weight
                    else:
                        # Use count of infected neighbors
                        infection_prob = 1 - (1 - beta) ** len(infected_neighbors)
                else:
                    infection_prob = beta  # Simple infection process
                
                # Update infection status
                if eventp < infection_prob:
                    actual_status[u] = 1
                

        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        self.actual_iteration += 1

        if node_status:
            return {
                "iteration": self.actual_iteration - 1,
                "status": delta.copy(),
                "node_count": node_count.copy(),
                "status_delta": status_delta.copy(),
            }
        else:
            return {
                "iteration": self.actual_iteration - 1,
                "status": {},
                "node_count": node_count.copy(),
                "status_delta": status_delta.copy(),
            }
