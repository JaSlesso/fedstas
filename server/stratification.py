'''
import numpy as np
from typing import List, Dict

def stratify_clients(gradients: List[np.ndarray], H: int, max_iter: int = 100) -> Dict[int, List[int]]:
    """
    Cluster clients into H strata using custom client stratification algorithm (Algorithm 2).
    
    Args:
        gradients (List[np.ndarray]): Reconstructed gradients Z_k for each client (each shape (d,))
        H (int): Number of strata (groups)
        max_iter (int): Maximum number of iterations for convergence

    Returns:
        Dict[int, List[int]]: Dictionary mapping stratum index to list of client indices
    """

    N = len(gradients)
    d =  gradients[0].shape[0]

    # Step 1: Initialize mu_i as H random client gradients
    rng = np.random.default_rng(seed=42)
    initial_indices = rng.choice(N, size=H, replace=False)
    mu = [gradients[i].copy() for i in initial_indices]

    # Initialize clusters
    strata = {h: [] for h in range(H)}
    converged = False
    iter_count = 0

    while not converged and iter_count < max_iter:
        iter_count += 1

        # Clear current clusters
        strata = {h: [] for h in range(H)}

        # Step 2: Assign each client to nearest cluster
        for k, Z_k in enumerate(gradients):
            distances = [np.linalg.norm(Z_k - mu_i) for mu_i in mu]
            h = int(np.argmin(distances)) # nearest stratum index
            strata[h].append(k)

        # Step 3: Update centers
        new_mu = []
        for h in range(H):
            if strata[h]: # avoid empty strata
                Z_h = np.stack([gradients[k] for k in strata[h]])
                new_mu_h = np.mean(Z_h, axis=0)
                new_mu.append(new_mu_h)
            else:
                # reinitialize to a random client if empty
                new_mu.append(gradients[rng.choice(N)].copy())

        # Step 4: Convergence check
        converged = all(np.allclose(mu[h], new_mu[h]) for h in range(H))
        mu = new_mu
    
    return strata


def compute_stratum_statistics(gradients: List[np.ndarray], strata: Dict[int, List[int]]) -> Dict[int, float]:
    """
    Compute the average L2 deviation from the center for each stratum (S_h).

    Args:
        gradients (List[np.ndarray]): List of reconstructed gradients Z_k (each shape (d,))
        strata (Dict[int, List[int]]): Mapping from stratum index h to list of client indices

    Returns:
        Dict[int, float]: Mapping from stratum index h to its variability S_h
    """
    S_h = {}
    for h, client_indices in strata.items():
        if not client_indices:
            S_h[h] = 0.0
            continue

        Z_h = np.stack([gradients[k] for k in client_indices])
        var = np.var(Z_h, axis=0)
        S_h[h] = np.mean(var)

    return S_h


def neyman_allocation(N_h: Dict[int, int], S_h: Dict[int, float], m: int) -> Dict[int, int]:
    """
    Allocate m samples across H strata using Neyman allocation.

    Args:
        N_h (Dict[int, int]): Number of clients in each stratum h
        S_h (Dict[int, float]): Variability S_h for each stratum h
        m (int): Total number of clients to sample

    Returns:
        Dict[int, int]: Number of clients to sample from each stratum h (m_h)
    """
    # Compute raw weights
    weights = {h: N_h[h] * S_h[h] for h in N_h}
    total_weight = sum(weights.values())

    if total_weight == 0 or np.isnan(total_weight):
        # Fallback to uniform allocation
        non_empty = [h for h in N_h if N_h[h] > 0]
        alloc = {h: 0 for h in N_h}
        for i, h in enumerate(non_empty[:m]):
            alloc[h] += 1
        return alloc

    # Allocate proportionally
    m_h = {h: int(np.floor(m * weights[h] / total_weight)) for h in N_h}

    # Correction step: add remaining clients (due to rounding) to strata with largest remainder
    remainder = m - sum(m_h.values())
    if remainder > 0:
        remainders = {
            h: (m * weights[h] / total_weight) - m_h[h]
            for h in N_h
        }
        top_up = sorted(remainders.items(), key=lambda x: -x[1])[:remainder]
        for h, _ in top_up:
            m_h[h] += 1
    
    return m_h


def importance_sample(stratum_clients: List[int], norms: List[float], m_h: int, replace: bool = False) -> List[int]:
    """
    Importance sample m_h clients from a stratum using gradient norms as probabilities.

    Args:
        stratum_clients (List[int]): Indices of clients in the stratum
        norms (List[float]): L2 norms of each client's gradient in the same order
        m_h (int): Number of clients to sample
        replace (bool): Whether to sample with replacement (default False)

    Returns:
        List[int]: Selected client indices (from stratum_clients)
    """
    assert len(stratum_clients) == len(norms), "Mismatch between clients and norm list"

    norm_array = np.array(norms)
    if np.sum(norm_array) == 0:
        # If all norms are zero, fall back to uniform sampling
        probs = np.ones(len(norm_array)) / len(norm_array)
    else:
        probs = norm_array / norm_array.sum()
    
    m_eff = min(m_h, len(stratum_clients))  # cap to available size

    selected_indices = np.random.choice(
        len(stratum_clients),
        size=m_eff,
        replace=replace,
        p=probs
    )

    return [stratum_clients[i] for i in selected_indices]
'''

import numpy as np
from typing import List, Dict

def stratify_clients(gradients: List[np.ndarray], H: int, max_iter: int = 100) -> Dict[int, List[int]]:
    """
    Cluster clients into H strata using custom client stratification algorithm (Algorithm 2).
    
    Args:
        gradients (List[np.ndarray]): Reconstructed gradients Z_k for each client (each shape (d,))
        H (int): Number of strata (groups)
        max_iter (int): Maximum number of iterations for convergence

    Returns:
        Dict[int, List[int]]: Dictionary mapping stratum index to list of client indices
    """

    N = len(gradients)
    d =  gradients[0].shape[0]

    # Step 1: Initialize mu_i as H random client gradients
    rng = np.random.default_rng(seed=42)
    initial_indices = rng.choice(N, size=H, replace=False)
    mu = [gradients[i].copy() for i in initial_indices]

    # Initialize clusters
    strata = {h: [] for h in range(H)}
    converged = False
    iter_count = 0

    while not converged and iter_count < max_iter:
        iter_count += 1

        # Clear current clusters
        strata = {h: [] for h in range(H)}

        # Step 2: Assign each client to nearest cluster
        for k, Z_k in enumerate(gradients):
            distances = [np.linalg.norm(Z_k - mu_i) for mu_i in mu]
            h = int(np.argmin(distances)) # nearest stratum index
            strata[h].append(k)

        # Step 3: Update centers
        new_mu = []
        for h in range(H):
            if strata[h]: # avoid empty strata
                Z_h = np.stack([gradients[k] for k in strata[h]])
                new_mu_h = np.mean(Z_h, axis=0)
                new_mu.append(new_mu_h)
            else:
                # reinitialize to a random client if empty
                new_mu.append(gradients[rng.choice(N)].copy())

        # Step 4: Convergence check
        converged = all(np.allclose(mu[h], new_mu[h]) for h in range(H))
        mu = new_mu
    
    return strata


def compute_stratum_statistics(gradients: List[np.ndarray], strata: Dict[int, List[int]]) -> Dict[int, float]:
    """
    Compute the average L2 deviation from the center for each stratum (S_h).

    Args:
        gradients (List[np.ndarray]): List of reconstructed gradients Z_k (each shape (d,))
        strata (Dict[int, List[int]]): Mapping from stratum index h to list of client indices

    Returns:
        Dict[int, float]: Mapping from stratum index h to its variability S_h
    """
    S_h = {}
    for h, client_indices in strata.items():
        if not client_indices:
            S_h[h] = 0.0
            continue

        Z_h = np.stack([gradients[k] for k in client_indices])
        var = np.var(Z_h, axis=0)
        S_h[h] = np.mean(var)

    return S_h


def neyman_allocation(N_h: Dict[int, int], S_h: Dict[int, float], m: int) -> Dict[int, int]:
    """
    Allocate m samples across H strata using Neyman allocation.

    Args:
        N_h (Dict[int, int]): Number of clients in each stratum h
        S_h (Dict[int, float]): Variability S_h for each stratum h
        m (int): Total number of clients to sample

    Returns:
        Dict[int, int]: Number of clients to sample from each stratum h (m_h)
    """
    # Compute raw weights
    weights = {h: N_h[h] * S_h[h] for h in N_h}
    total_weight = sum(weights.values())

    if total_weight == 0 or np.isnan(total_weight):
        # Fallback to uniform allocation
        non_empty = [h for h in N_h if N_h[h] > 0]
        alloc = {h: 0 for h in N_h}
        for i, h in enumerate(non_empty[:m]):
            alloc[h] += 1
        return alloc

    # Allocate proportionally
    m_h = {h: int(np.floor(m * weights[h] / total_weight)) for h in N_h}

    # Correction step: add remaining clients (due to rounding) to strata with largest remainder
    remainder = m - sum(m_h.values())
    if remainder > 0:
        remainders = {
            h: (m * weights[h] / total_weight) - m_h[h]
            for h in N_h
        }
        top_up = sorted(remainders.items(), key=lambda x: -x[1])[:remainder]
        for h, _ in top_up:
            m_h[h] += 1
    
    return m_h


def importance_sample(stratum_clients: List[int], norms: List[float], m_h: int, replace: bool = False) -> List[int]:
    """
    Importance sample m_h clients from a stratum using gradient norms as probabilities.

    Args:
        stratum_clients (List[int]): Indices of clients in the stratum
        norms (List[float]): L2 norms of each client's gradient in the same order
        m_h (int): Number of clients to sample
        replace (bool): Whether to sample with replacement (default False)

    Returns:
        List[int]: Selected client indices (from stratum_clients)
    """
    assert len(stratum_clients) == len(norms), "Mismatch between clients and norm list"

    norm_array = np.array(norms)
    if np.sum(norm_array) == 0:
        # If all norms are zero, fall back to uniform sampling
        probs = np.ones(len(norm_array)) / len(norm_array)
    else:
        probs = norm_array / norm_array.sum()
    
    m_eff = min(m_h, len(stratum_clients))  # cap to available size

    selected_indices = np.random.choice(
        len(stratum_clients),
        size=m_eff,
        replace=replace,
        p=probs
    )

    return [stratum_clients[i] for i in selected_indices]

    
