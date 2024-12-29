from ..DiffusionModel import DiffusionModel
import numpy as np
import future.utils

__author__ = "Giulio Rossetti"
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"


class SIRModel(DiffusionModel):
    """
    Model Parameters to be specified via ModelConfig

    :param beta: The infection rate (float value in [0,1])
    :param gamma: The recovery rate (float value in [0,1])
    """

    def __init__(self, graph, seed=None):
        """
        Model Constructor

        :param graph: A networkx graph object
        """
        super(self.__class__, self).__init__(graph, seed)
        self.available_statuses = {"Susceptible": 0, "Infected": 1, "Removed": 2}

        self.parameters = {
            "model": {
                "beta": {"descr": "Infection rate", "range": [0, 1], "optional": False},
                "gamma": {"descr": "Recovery rate", "range": [0, 1], "optional": False},
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

        self.active = []
        self.name = "SIR"

    def iteration(self, node_status=True):
        """
        Execute a single model iteration

        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        self.clean_initial_status(self.available_statuses.values())

        actual_status = {
            node: nstatus for node, nstatus in future.utils.iteritems(self.status)
        }
        self.active = [
            node
            for node, nstatus in future.utils.iteritems(self.status)
            if nstatus != self.available_statuses["Susceptible"]
        ]

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

        for u in self.active:

         legacy=False
         if not legacy:

            if self.status[u] != 1:  # Only process infected nodes
                continue

            # Get susceptible neighbors based on graph direction
            if self.graph.directed:
                neighbors = self.graph.successors(u)
            else:
                neighbors = self.graph.neighbors(u)
            
            susceptible_neighbors = [v for v in neighbors if self.status[v] == 0]

            # Try to infect each susceptible neighbor
            beta = self.params["model"]["beta"]
            use_tp = self.params["model"]["tp_rate"] == 1


            for v in susceptible_neighbors:
                eventp = np.random.random_sample()
                
                if use_tp and self.graph_has_weights:
                    # Get edge weights
                    all_weights = self.graph.get_edge_attributes("weight")
                    edge_weight = all_weights[(u,v)]  # Note: switched from (v,u) to (u,v) since u is infecting v (unlike this SIModel implementation)
                    infection_prob = 1 - (1 - beta) ** edge_weight
                else:
                    # Note: A bit unclear on how this is not using tp originally anyway, PH keeps it for compatibility.
                    infection_prob = beta
                 
                if eventp < infection_prob:
                    actual_status[v] = 1
            
            # Check if infected node recovers
            gamma = self.params["model"]["gamma"]
            if np.random.random_sample() < gamma:
                actual_status[u] = 2
            # End
         else:
            
            u_status = self.status[u]

            if u_status == 1:

                if self.graph.directed:
                    susceptible_neighbors = [
                        v for v in self.graph.successors(u) if self.status[v] == 0
                    ]
                else:
                    susceptible_neighbors = [
                        v for v in self.graph.neighbors(u) if self.status[v] == 0
                    ]
                for v in susceptible_neighbors:
                    eventp = np.random.random_sample()
                    if eventp < self.params["model"]["beta"]:
                        actual_status[v] = 1

                eventp = np.random.random_sample()
                if eventp < self.params["model"]["gamma"]:
                    actual_status[u] = 2

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
