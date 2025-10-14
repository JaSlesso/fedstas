import torch
from typing import Dict, List

def aggregate_models(models_by_stratum: Dict[int, List[torch.nn.Module]], N_h: Dict[int, int], m_h: Dict[int, int]) -> torch.nn.Module:
    """
    Aggregate client models weighted by stratum sizes and sampling proportions.

    Args:
        models_by_stratum (Dict[int, List[nn.Module]]): Models from each stratum h
        N_h (Dict[int, int]): Total number of clients in each stratum
        m_h (Dict[int, int]): Number of clients actually sampled from each stratum

    Returns:
        nn.Module: The aggregated global model
    """
    # Get parameter structure from the first model
    #example_model = next(iter(next(iter(models_by_stratum.values())))) # model from first stratum
    # Get parameter structure from the first non-empty stratum
    example_model = None
    for stratum_models in models_by_stratum.values():
        if stratum_models:  # Find first non-empty stratum
            example_model = stratum_models[0]
            break
    
    if example_model is None:
        raise ValueError("No models to aggregate - all strata are empty!")
    
    global_state = {k: torch.zeros_like(v) for k, v in example_model.state_dict().items()}
    total_clients = sum(N_h.values())

    for h, client_models in models_by_stratum.items():
        if not client_models or m_h[h] == 0:
            continue

        # Average model in stratum
        stratum_sum = {k: torch.zeros_like(v) for k, v in example_model.state_dict().items()}
        for model in client_models:
            for k, v in model.state_dict().items():
                stratum_sum[k] += v
        
        #stratum_avg = {k: v / m_h[h] for k, v in stratum_sum.items()}
        # Use actual number of models (some clients may have been skipped due to 0 samples)
        actual_m_h = len(client_models)
        stratum_avg = {k: v / actual_m_h for k, v in stratum_sum.items()}

        # Weighted by N_h / N
        weight = N_h[h] / total_clients
        for k in global_state:
            global_state[k] += weight * stratum_avg[k]

    # Load aggregated weights into a new model
    new_model = type(example_model)() # assumes model can be constructed with no args
    new_model.load_state_dict(global_state)
    return new_model
