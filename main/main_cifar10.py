# cifar10.py — Run 3 methods (FedSTS / FedSTaS / FedSTaS+DP) on CIFAR-10


import os, sys, random, argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from collections import defaultdict
import matplotlib.pyplot as plt

# --- repo model import  ---
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.cifar10 import create_model

from server.coordinator import FedSTaSCoordinator


# ----------------------------
parser = argparse.ArgumentParser(description="FedSTS vs FedSTaS on CIFAR-10")
parser.add_argument("--beta", type=float, default=0.01, help="Dirichlet concentration (smaller => more non-IID)")
parser.add_argument("--epsilon", type=float, default=None, help="DP budget epsilon (None => no DP for the DP run)")
parser.add_argument("--M", type=int, default=300, help="DP clip cap M")
parser.add_argument("--rounds", type=int, default=300, help="Federated rounds")
parser.add_argument("--model", type=str, default="fast_cnn", help="Model type for create_model(...)")
parser.add_argument("--clients", type=int, default=100, help="Total clients")
parser.add_argument("--h", type=int, default=10, help="Number of strata")
parser.add_argument("--clients_per_round", type=int, default=10, help="Clients sampled per round")
parser.add_argument("--n_star", type=int, default=2500, help="Target data per round (None/0 => FedSTS)")
parser.add_argument("--epochs", type=int, default=2, help="Local epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Local batch size")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
parser.add_argument("--seed", type=int, default=562, help="RNG seed")
parser.add_argument("--verbose", action="store_true", help="More logs")
parser.add_argument("--csv", type=str, default="cifar10_beta_eps_results.csv", help="Output CSV filename")
parser.add_argument("--iid", action="store_true", help="Use IID data distribution instead of Dirichlet")
args = parser.parse_args()

# ----------------------------
# 1) Repro & device
# ----------------------------
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ----------------------------
# 2) CIFAR-10 IID & Dirichlet split 
# ----------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

cifar_train = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
cifar_test  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

def split_cifar10_dirichlet(dataset, num_clients=100, beta=0.01):
    """Split CIFAR-10 using Dirichlet distribution for non-IID data"""
    labels = np.array(dataset.targets)
    indices = np.arange(len(dataset))
    class_indices = [indices[labels == c] for c in range(10)]
    client_indices = defaultdict(list)
    for c in range(10):
        props = np.random.dirichlet(np.repeat(beta, num_clients))
        cuts = (np.cumsum(props) * len(class_indices[c])).astype(int)[:-1]
        splits = np.split(class_indices[c], cuts)
        for i, idx in enumerate(splits):
            client_indices[i].extend(idx.tolist())
    return [Subset(dataset, sorted(idxs)) for i, idxs in sorted(client_indices.items())]

def split_cifar10_iid(dataset, num_clients=100):
    """Split CIFAR-10 uniformly for IID data distribution"""
    total_samples = len(dataset)
    samples_per_client = total_samples // num_clients
    remainder = total_samples % num_clients
    
    # Shuffle all indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    client_indices = []
    start_idx = 0
    
    for i in range(num_clients):
        # Add one extra sample to first 'remainder' clients
        client_size = samples_per_client + (1 if i < remainder else 0)
        end_idx = start_idx + client_size
        
        client_indices.append(Subset(dataset, indices[start_idx:end_idx]))
        start_idx = end_idx
    
    return client_indices

# Create client datasets based on IID flag
if args.iid:
    client_datasets = split_cifar10_iid(cifar_train, num_clients=args.clients)
    print(f"Created {len(client_datasets)} client datasets (IID distribution).")
    print(f"Average samples per client: {np.mean([len(ds) for ds in client_datasets]):.1f}")
else:
    client_datasets = split_cifar10_dirichlet(cifar_train, num_clients=args.clients, beta=args.beta)
    print(f"Created {len(client_datasets)} client datasets (β={args.beta}).")
    print(f"Average samples per client: {np.mean([len(ds) for ds in client_datasets]):.1f}")

# ----------------------------
# 3) Config & helpers
# ----------------------------
def alpha_from_eps(eps, M):
    # α = (e^ε - 1) / (e^ε + M - 2)
    return float((np.exp(eps) - 1.0) / (np.exp(eps) + (M - 2)))

BASE_CFG = dict(
    H=args.h,
    d_prime=5,
    restratify_every=10,
    clients_per_round=args.clients_per_round,
    M=args.M,
    n_star=args.n_star,          
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    weight_decay=args.weight_decay,
    verbose=args.verbose,
)

def make_coordinator(cfg_overrides=None):
    cfg = BASE_CFG.copy()
    if cfg_overrides: cfg.update(cfg_overrides)

    # derive alpha when epsilon is given
    eps = cfg.get("epsilon", None)
    if eps is not None:
        cfg["alpha"] = alpha_from_eps(eps, cfg["M"])

    cfg["use_data_sampling"] = (cfg.get("n_star", None) not in (None, 0))

    return FedSTaSCoordinator(
        global_model=create_model(args.model, num_classes=10),
        client_datasets=client_datasets,
        test_dataset=cifar_test,
        config=cfg,
        device=DEVICE,
        verbose=cfg.get("verbose", False),
    )

def run_once(label, cfg_overrides=None, num_rounds=100):
    print(f"\n=== Running: {label} ===")
    coord = make_coordinator(cfg_overrides)
    coord.run(num_rounds=num_rounds)
    return np.array(coord.validation_curve, dtype=float), np.array(coord.validation_loss_curve, dtype=float)

# ----------------------------
# 4) Define the 3 runs
# ----------------------------
runs = [
    ("FedSTS",          dict(n_star=None,               epsilon=None)),          
    ("FedSTaS (no-DP)", dict(n_star=BASE_CFG["n_star"], epsilon=None)),           
    (f"FedSTaS (ε={args.epsilon})", dict(n_star=BASE_CFG["n_star"], epsilon=args.epsilon)),  
]

# If epsilon is not provided, skip the DP run to avoid confusion
if args.epsilon is None:
    runs = runs[:2]

# ----------------------------
# 5) Execute & collect
# ----------------------------
results_acc = {}
results_loss = {}
for label, over in runs:
    acc_curve, loss_curve = run_once(label, over, num_rounds=args.rounds)
    results_acc[label] = acc_curve
    results_loss[label] = loss_curve

# ----------------------------
# 6) Plot two separate figures 
# ----------------------------
# Set Statistical Sinica style
plt.style.use('default')  # Start with default
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 2,
    'axes.linewidth': 1,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7
})

# Define colors for the three methods
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
linestyles = ['-', '--', '-.']
markers = ['o', 's', '^']

# Plot 1: Validation Accuracy
plt.figure(figsize=(10, 6))
for i, (label, curve) in enumerate(results_acc.items()):
    plt.plot(range(1, len(curve)+1), curve, 
             color=colors[i], linestyle=linestyles[i], marker=markers[i],
             label=label, linewidth=2, markersize=4, markevery=5)

plt.title('Validation Accuracy', fontsize=16, fontweight='bold')
plt.xlabel('Round', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, alpha=0.7)
plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.show()

# Plot 2: Validation Loss
plt.figure(figsize=(10, 6))
for i, (label, curve) in enumerate(results_loss.items()):
    plt.plot(range(1, len(curve)+1), curve, 
             color=colors[i], linestyle=linestyles[i], marker=markers[i],
             label=label, linewidth=2, markersize=4, markevery=5)

plt.title('Validation Loss', fontsize=16, fontweight='bold')
plt.xlabel('Round', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.7)
plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.show()

# ----------------------------
# 7) Save CSV
# ----------------------------
import pandas as pd
rows = []
for label in results_acc.keys():
    acc_curve = results_acc[label]
    loss_curve = results_loss[label]
    for r, (acc, loss) in enumerate(zip(acc_curve, loss_curve), start=1):
        rows.append({
            "method": label, "round": r, "accuracy": float(acc), "loss": float(loss),
            "beta": args.beta, "epsilon": args.epsilon, "M": args.M,
            "model": args.model, "clients": args.clients,
            "H": args.h, "clients_per_round": args.clients_per_round, "n_star": args.n_star
        })
pd.DataFrame(rows).to_csv(args.csv, index=False)
print("Saved CSV:", os.path.abspath(args.csv))

'''
%cd /content/fedstas
!python main/main_cifar10.py --beta 0.01 --epsilon 3 --rounds 100 --model fast_cnn --clients 100 --h 10 --clients_per_round 10 --n_star 2500 --M 300
!python main/main_cifar10.py --iid --epsilon 3 --rounds 100 --model fast_cnn --clients 100 --h 10 --clients_per_round 10 --n_star 2500 --M 300
'''
