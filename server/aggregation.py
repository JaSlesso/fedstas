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



# server/aggregation_weighted.py
import copy
import torch
from typing import Dict, List

def _model_to_vec(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def _vec_to_model(vec: torch.Tensor, like_model: torch.nn.Module) -> torch.nn.Module:
    out = copy.deepcopy(like_model)
    offset = 0
    with torch.no_grad():
        for p in out.parameters():
            n = p.numel()
            p.data.copy_(vec[offset:offset+n].view_as(p))
            offset += n
    return out

def aggregate_models_weighted(
    models_by_stratum: Dict[int, List[torch.nn.Module]],
    counts_by_stratum: Dict[int, List[int]],
    N_h: Dict[int, int],
) -> torch.nn.Module:
    """
    Two-stage stratified aggregation:
      (i) within each stratum h, compute a client-weighted mean using n_used_k
      (ii) across strata, keep the FedSTaS outer weighting by N_h (population size)

    Produces: w_{t+1} = (1/N) * sum_h N_h * w̄_h,
      where w̄_h = (sum_{k in h} n_used_k * w_k) / (sum_{k in h} n_used_k).
    This mirrors FedAvg inside each stratum and FedSTaS across strata.
    """
    # find a template model
    template = None
    for ms in models_by_stratum.values():
        if ms:
            template = ms[0]
            break
    if template is None:
        raise ValueError("No models to aggregate.")

    device = next(template.parameters()).device
    stratum_means = {}
    total_N = float(sum(N_h.values()))

    # (i) within-stratum weighted mean by actual examples used
    for h, models in models_by_stratum.items():
        counts = counts_by_stratum.get(h, [])
        if not models or not counts:
            continue

        assert len(models) == len(counts), f"len mismatch in stratum {h}"
        num = 0.0
        den = 0.0
        for mdl, c in zip(models, counts):
            if c <= 0:
                continue
            num_vec = _model_to_vec(mdl).to(device) * float(c)
            num = num_vec if den == 0.0 else (num + num_vec)
            den += float(c)

        if den > 0.0:
            stratum_means[h] = num / den  # w̄_h
        else:
            # fallback: simple mean if den==0
            mean_vec = sum((_model_to_vec(m).to(device) for m in models)) / len(models)
            stratum_means[h] = mean_vec

    # (ii) across-strata: weight each stratum mean by N_h (as in FedSTaS)
    acc = None
    for h, mean_vec in stratum_means.items():
        weight = float(N_h.get(h, 0))
        if weight <= 0:
            continue
        term = mean_vec * weight
        acc = term if acc is None else (acc + term)

    if acc is None:
        # nothing aggregated; return template unchanged
        return template

    global_vec = acc / total_N
    return _vec_to_model(global_vec, template)
