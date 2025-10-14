import torch
import numpy as np
from typing import List, Dict
from client.compression import compress_gradient, decompress_gradient
from client.trainer import local_train, get_raw_update
from client.privacy import clip_and_fake
from client.sampling import sample_uniform_data
from server.stratification import (
    stratify_clients, compute_stratum_statistics,
    neyman_allocation, importance_sample
)
from server.privacy import estimate_total_sample_size
from server.aggregation import aggregate_models
from utils.evals import evaluate_model
import torch.nn as nn ###
from torch.utils.data import DataLoader, Subset
class FedSTaSCoordinator:
    def __init__(
            self,
            global_model: torch.nn.Module,
            client_datasets: List[torch.utils.data.Dataset],
            test_dataset,
            config: Dict,
            device: str = "cpu",
            verbose: bool = True
    ):
        self.global_model = global_model
        self.client_datasets = client_datasets
        self.test_dataset = test_dataset
        self.validation_curve = []
        self.validation_loss_curve = []
        self.device = device
        self.config = config
        self.num_clients = len(client_datasets)
        self.verbose = verbose
        self.restratify_every = self.config.get("restratify_every", 1)
        self.cached_strata = None
        self.cached_standardized = None
        self.cached_S_h = None
        self.cached_N_h = None
        self.cached_m_h = None
        ####added proxy_frac
        self.proxy_frac = float(self.config.get("proxy_frac", 0.10))      
        self.proxy_cap  = int(self.config.get("proxy_cap", 128))          
        self.stratify_device = str(self.config.get("stratify_device", "cpu"))  
        # project proxy vectors to  d_prime 
        self.proj_dim = int(self.config.get("proj_dim", self.config.get("d_prime", 5)))

        # cache: standardized proxy matrix for all clients (used for norms, etc.)
        self.cached_G = None  # shape [N, d']

    def run(self, num_rounds: int):
        for round_idx in range(num_rounds):
            print(f"\n=== Round {round_idx + 1} ===")
            '''
'''
            # === Step 1–3: Optional Re-stratification ===
            if round_idx == 0 or round_idx % self.restratify_every == 0:
                if self.verbose:
                    print("\n[Stratification] (Recomputing gradients + clusters)")

                compressed_grads = []
                for i, dataset in enumerate(self.client_datasets):
                    raw_grad = get_raw_update(self.global_model, dataset, device=self.device)
                    centroids, indices = compress_gradient(raw_grad, self.config["d_prime"])
                    compressed_grads.append((centroids, indices))
                    if self.verbose and i < 5:
                        print(f"  Client {i}: ||raw_grad|| = {np.linalg.norm(raw_grad):.4f}")

                # Step 2: Reconstruct and Standardize
                reconstructed = [decompress_gradient(c, i) for (c, i) in compressed_grads]
                grad_matrix = np.stack(reconstructed)
                mu = grad_matrix.mean(axis=0)
                sigma = grad_matrix.std(axis=0) + 1e-8
                standardized = [(g - mu) / sigma for g in reconstructed]

                # Step 3: Stratify
                self.cached_strata = stratify_clients(standardized, self.config["H"])
                self.cached_S_h = compute_stratum_statistics(standardized, self.cached_strata)
                self.cached_N_h = {h: len(c) for h, c in self.cached_strata.items()}
                self.cached_m_h = neyman_allocation(
                    self.cached_N_h, self.cached_S_h, self.config["clients_per_round"]
                )
            else:
                if self.verbose:
                    print("\n[Stratification] (Using cached strata and allocations)")
            '''
'''

            # === Step 1–3: Optional Re-stratification ===
            if round_idx == 0 or round_idx % self.restratify_every == 0:
                if self.verbose:
                    print("\n[Stratification] (Recomputing proxy grads + MiniBatchKMeans)")

                # Build proxy gradients for ALL clients (cheap: one tiny batch each, on CPU)
                all_client_indices = list(range(self.num_clients))
                G_all = build_client_proxies(
                    selected_clients=all_client_indices,
                    global_model=self.global_model,
                    client_datasets=self.client_datasets,
                    device=self.stratify_device,       # "cpu" recommended on Colab
                    frac=self.proxy_frac,              # ~10% of local data
                    cap=self.proxy_cap,                # but capped at 128 examples
                    last_layer_only=True,              # tiny, stable proxy (final layer grad)
                    proj_dim=self.proj_dim,            # d' (matches your d_prime)
                    proj_seed=0,
                    autocast_dtype=None                # set torch.float16 if you move this to GPU
                )   # shape [N, d']

                # Standardize (z-score) so clustering isn’t scale-dominated
                mu = G_all.mean(axis=0)
                sigma = G_all.std(axis=0) + 1e-8
                G_std = (G_all - mu) / sigma
                self.cached_G = G_std                  # keep for Step 4 norms

                # Cluster ALL clients into H strata with MiniBatchKMeans (fast CPU)
                strata_local = stratify_clients_mbkm(
                    G_std, H=self.config["H"], batch_size=256, max_iter=50, seed=42
                )
                # map local indices (0..N-1) back to global client ids
                self.cached_strata = {h: [all_client_indices[i] for i in idxs]
                                      for h, idxs in strata_local.items()}

                # Variability per stratum (for Neyman allocation)
                self.cached_S_h = compute_stratum_variability(G_std, self.cached_strata)
                self.cached_N_h = {h: len(c) for h, c in self.cached_strata.items()}
                self.cached_m_h = neyman_allocation(
                    self.cached_N_h, self.cached_S_h, self.config["clients_per_round"]
                )
            else:
                if self.verbose:
                    print("\n[Stratification] (Using cached strata and allocations)")
            

            
            # Assign from cache
            strata = self.cached_strata
            S_h = self.cached_S_h
            N_h = self.cached_N_h
            m_h = self.cached_m_h
            
            if self.verbose:
                print("\n[Stratification]")
                for h, clients in strata.items():
                    print(f"  Stratum {h}: N_h = {len(clients)}, S_h = {S_h[h]:.4f}, m_h = {m_h[h]}")

            # Step 4: Sample clients via importance sampling
            selected_clients_by_stratum = {}
            if self.verbose:
                print("\n[Client Selection]")
            for h, client_indices in strata.items():
                if not client_indices or m_h[h] == 0:
                    continue
                #norms = [np.linalg.norm(reconstructed[k]) for k in client_indices]
                norms = [np.linalg.norm(self.cached_G[k]) for k in client_indices]
                selected = importance_sample(client_indices, norms, m_h[h])
                selected_clients_by_stratum[h] = selected
                if self.verbose:
                    print(f"  Stratum {h}: selected {selected}")
            
            # Step 5: Collect privatized sample counts
            responses = []
            if self.verbose:
                print("\n[Sample Size Reporting]")
            for h, clients in selected_clients_by_stratum.items():
                for k in clients:
                    n_k = len(self.client_datasets[k])
                    r_k = clip_and_fake(n_k, self.config["M"], self.config["alpha"])
                    responses.append(r_k)
                    if self.verbose:
                        print(f"  Client {k}: n_k = {n_k}, r_k = {r_k}")
            
            # Step 6: Estimate total sample count
            n_tilde = estimate_total_sample_size(responses, self.config["alpha"], self.config["M"])
            p = min(self.config["n_star"] / n_tilde, 1.0)
            print(f"Estimated ñ = {n_tilde:.2f}, using sampling ratio p = {p:.4f}")

            # Step 7: Train selected clients
            models_by_stratum = {}
            if self.verbose:
                print("\n[Local Training]")
            for h, clients in selected_clients_by_stratum.items():
                local_models = []
                for k in clients:
                    model_copy = self.global_model.to(self.device)
                    subset = sample_uniform_data(
                        self.client_datasets[k], p, seed=round_idx
                    )
                    if self.verbose:
                        print(f"  Client {k}: training on {len(subset)} samples")
                    if len(subset) == 0:
                        if self.verbose:
                            print(f"  Client {k}: skipped (0 samples)")
                        continue

                    updated_model = local_train(
                        model=model_copy,
                        dataset=subset,
                        epochs=self.config["epochs"],
                        batch_size=self.config["batch_size"],
                        lr=self.config["lr"],
                        sample_fraction=1.0,
                        weight_decay=self.config.get("weight_decay", 0.0),
                        device=self.device
                    )
                    local_models.append(updated_model)
                models_by_stratum[h] = local_models
            
            # Step 8: Aggregate updates
            self.global_model = aggregate_models(models_by_stratum, N_h, m_h)
            print("Aggregated global model updated.")

            # Step 9: Evaluate (optional)
            if self.test_dataset is not None:
                acc, val_loss = evaluate_model(self.global_model, self.test_dataset, device=self.device)
                print(f"Validation accuracy: {acc*100:.3f}% | Loss: {val_loss:.4f}")
                self.validation_curve.append(acc)
                self.validation_loss_curve.append(val_loss)
