import torch
import copy
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



def _model_to_vec(model: torch.nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def _vec_to_model(vec: torch.Tensor, like_model: torch.nn.Module) -> torch.nn.Module:
    """Reconstruct a model from a flattened vector using like_model as template."""
    out = copy.deepcopy(like_model)
    offset = 0
    with torch.no_grad():
        for p in out.parameters():
            num = p.numel()
            p.data.copy_(vec[offset:offset+num].view_as(p))
            offset += num
    return out


def aggregate_models_weighted(
    models_by_stratum: Dict[int, List[torch.nn.Module]], 
    used_counts_by_stratum: Dict[int, List[int]]
) -> torch.nn.Module:
    """
    FedAvg-style weighted aggregation: weight each local model by actual local examples used.
    
    Args:
        models_by_stratum: Models from each stratum h
        used_counts_by_stratum: Number of samples each client actually trained on
    
    Returns:
        Aggregated global model
    """
    # Get device from first available model
    device = None
    template_model = None
    for models in models_by_stratum.values():
        if models:
            template_model = models[0]
            device = next(template_model.parameters()).device
            break
    
    if template_model is None:
        raise ValueError("No models to aggregate - all strata are empty!")
    
    acc_vec = None
    total_samples = 0
    
    for h, models in models_by_stratum.items():
        counts = used_counts_by_stratum.get(h, [])
        if not models:
            continue
        
        assert len(models) == len(counts), (
            f"Mismatch between models ({len(models)}) and counts ({len(counts)}) in stratum {h}"
        )
        
        for mdl, cnt in zip(models, counts):
            if cnt <= 0:
                continue
            
            # Weight this model's parameters by number of samples it trained on
            v = _model_to_vec(mdl).to(device) * float(cnt)
            acc_vec = v if acc_vec is None else (acc_vec + v)
            total_samples += cnt
    
    if total_samples == 0:
        # Fallback: unweighted average if something went wrong
        all_models = [m for ms in models_by_stratum.values() for m in ms]
        if not all_models:
            raise ValueError("No models with positive sample counts!")
        avg = sum((_model_to_vec(m).to(device) for m in all_models)) / len(all_models)
        return _vec_to_model(avg, template_model)
    
    # Compute weighted average
    avg = acc_vec / float(total_samples)
    return _vec_to_model(avg, template_model)
