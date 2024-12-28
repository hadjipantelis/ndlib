import pytest
import networkx as nx
import numpy as np
import pandas as pd
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc

def get_tiny_weighted_graph(directed=True, weight=1):
    """
    Create a tiny weighted graph.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(0, 2, weight=weight)
    return G

@pytest.mark.parametrize("beta, weight", [
    (0.25, 15),
    (0.25, 1.5),
    (0.25, 0.15),
    (0.025, 15),
    (0.025, 1.5),
    # (0.025, 0.15), # Small number require more reps
])
def test_si_model(beta, weight):
    """
    Parametrized test for the SI model with given beta and weight.
    """
    reps = 5_000
    iteration_ones = []
    for _ in range(reps):  
        modelA = ep.SIModel(get_tiny_weighted_graph(weight=weight))
        # Model Configuration
        cfgA = mc.Configuration() 
        cfgA.add_model_parameter('beta', beta)
        cfgA.add_model_initial_configuration("Infected", [0])
        modelA.set_initial_status(cfgA)
        
        # Simulation execution
        iterationsA = modelA.iteration_bunch(2)
        iteration_ones.append(iterationsA[1]["status"])

    # Compute ground truth
    ground_truth = np.sort((
        (1 - (1 - beta) ** 1) - (1 - (1 - beta) ** weight) * (1 - (1 - beta) ** 1),
        (1 - (1 - beta) ** weight) * (1 - (1 - beta) ** 1),
        (1 - (1 - (1 - beta) ** weight)) * (1 - (1 - (1 - beta) ** 1)),
        (1 - (1 - beta) ** weight) - (1 - (1 - beta) ** weight) * (1 - (1 - beta) ** 1),
    )) 
    
    # Validate results
    np.testing.assert_allclose(
        np.sort(pd.Series(iteration_ones).value_counts().values / len(iteration_ones)),
        ground_truth,
        atol=0.005, rtol=1
    )
