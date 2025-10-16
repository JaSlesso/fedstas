import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import SGD, Adam, AdamW
from typing import Dict, Any

from client.sampling import sample_uniform_data

def local_train(
        model: Module,
        dataset: torch.utils.data.Dataset,
        epochs: int,
        batch_size: int,
        lr: float,
        sample_fraction: float,
        weight_decay: float = 0.0,
        device: str = "cpu",
        loss_fn = None,
        optimizer_type: str = "sgd",
        momentum: float = 0.9,
        use_cosine_decay: bool = True,
        min_samples_threshold: int = None
) -> Module:
    """
    Perform local training on a uniformly sampled subset of the dataset.

    Args:
        model (Module): The global model sent by the server
        dataset (Dataset): Full local dataset for the client
        epochs (int): Number of local training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate for optimizer
        sample_fraction (float): Fraction of local data to train on (from server)
        weight_decay (float): L2 regularization coefficient (default: 1e-4)
        device (str): 'cpu' or 'cuda'
        loss_fn (callable, optional): Loss function (defaults to CrossEntropyLoss)
        optimizer_type (str): Optimizer type: 'sgd', 'adam', or 'adamw' (default: 'sgd')
        momentum (float): Momentum for SGD optimizer (default: 0.9)
        use_cosine_decay (bool): Whether to use cosine annealing LR decay (default: True)
        min_samples_threshold (int): Minimum samples for training; if None, defaults to batch_size

    Returns:
        Module: Updated local model after training
    """
    model = model.to(device)
    model.train()

    # Step 1: Sample uniformly from the local dataset
    subset = sample_uniform_data(dataset, sample_fraction)
    #loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    
    n_samples = len(subset)
    
    # Step 2: Guard against ultra-tiny clients 
    if min_samples_threshold is None:
        min_samples_threshold = batch_size  # Default: at least 1 full batch
    
    if n_samples < min_samples_threshold:
        # Ultra-tiny client: return model unchanged (skip local training)
        # This prevents poisoning from clients with n=3-10 samples
        return model
    
    # Step 3: Adjust batch size for small clients
    # If n_samples < 2*batch_size, use smaller batches to get at least 2 batches per epoch
    effective_batch_size = batch_size
    if n_samples < 2 * batch_size:
        effective_batch_size = max(1, n_samples // 4)  # At least 4 batches if possible
    
    loader = DataLoader(subset, batch_size=effective_batch_size, shuffle=True)
    

    # Step 2: Set up optimizer with momentum and proper weight decay
    if optimizer_type.lower() == "sgd":
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # fallback to Adam for backward compatibility
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Step 3: Set up cosine annealing LR scheduler to reduce drift
    scheduler = None
    if use_cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    criterion = loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    # Step 4: Train with LR decay per epoch
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
    
    return model


def get_model_gradient(old_model: torch.nn.Module, new_model: torch.nn.Module) -> torch.Tensor:
    """
    Flatten and return the model difference (pseudo-gradient) as a single 1D tensor.

    Args:
        old_model (Module): The original global model before training
        new_model (Module): The locally updated model

    Returns:
        Tensor: Flattened gradient-like vector (1D tensor)
    """
    diffs = []
    for old_param, new_param in zip(old_model.parameters(), new_model.parameters()):
        diffs.append((new_param.data - old_param.data).flatten())
    return torch.cat(diffs)

def get_raw_update(model, dataset, device="cpu"):
    """
    Compute raw gradient vector from one batch of data.
    """
    model = model.to(device)
    model.train()
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    model.zero_grad()
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, y)
    loss.backward()

    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.flatten().detach().clone())
        else:
            grads.append(torch.zeros_like(p.data.flatten()))
    return torch.cat(grads).cpu().numpy()
