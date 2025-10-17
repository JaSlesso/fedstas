# cifar10.py — Run 3 methods (FedSTS / FedSTaS / FedSTaS+DP) on CIFAR-10


import os, sys, random, argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from collections import defaultdict

# --- repo model import  ---
import sys
import os

# Debug: Print current working directory and file location
print("Current working directory:", os.getcwd())
print("Script location:", __file__)
print("Script directory:", os.path.dirname(__file__))

# Get the absolute path to the repository root
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print("Repository root:", repo_root)
print("Contents of repo root:", os.listdir(repo_root))

# Add the repository root to Python path
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print("Python path:", sys.path[:3])  # Show first 3 entries

# Now import the modules
from model.cifar10 import create_model
from server.coordinator import FedSTaSCoordinator
import importlib
import client.trainer
import server.coordinator
import model.cifar10
importlib.reload(client.trainer)
importlib.reload(server.coordinator)
importlib.reload(model.cifar10)


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
parser.add_argument("--verbose",default=True, action="store_true", help="More logs")
parser.add_argument("--csv", type=str, default="cifar10_beta_eps_results.csv", help="Output CSV filename")
parser.add_argument("--iid", action="store_true", help="Use IID data distribution instead of Dirichlet")
parser.add_argument("--method", type=str, default="all", choices=["all", "fedsts", "fedstas_no_dp", "fedstas_dp"], 
                    help="Which method to run: all (default), fedsts, fedstas_no_dp, or fedstas_dp")
parser.add_argument("--clustering", type=str, default="gpu", choices=["minibatch", "gpu"],
                    help="Clustering method: minibatch (CPU, default) or gpu (cuML)")
parser.add_argument("--proxy_frac", type=float, default=0.30, help="Fraction of client data for proxy gradient (default: 0.30 = 30%%)")
parser.add_argument("--proxy_cap", type=int, default=512, help="Max samples for proxy gradient computation (default: 128)")
parser.add_argument("--d_prime", type=int, default=50, help="Projection dimension for stratification (default: 5)")
parser.add_argument("--restratify_every", type=int, default=20, help="Re-stratify every N rounds (default: 20)")
parser.add_argument("--optimizer_type", type=str, default="sgd", choices=["sgd", "adam", "adamw"],
                    help="Optimizer type: sgd (default), adam, or adamw")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer (default: 0.9)")
parser.add_argument("--use_cosine_decay", action="store_true", default=True, 
                    help="Use cosine annealing LR decay (default: True)")
parser.add_argument("--no_cosine_decay", dest="use_cosine_decay", action="store_false",
                    help="Disable cosine annealing LR decay")
parser.add_argument("--min_samples_threshold", type=int, default=None,
                    help="Minimum samples for local training; clients with fewer samples are skipped (default: batch_size)")
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
    d_prime=args.d_prime,
    restratify_every=args.restratify_every,
    clients_per_round=args.clients_per_round,
    M=args.M,
    n_star=args.n_star,          
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    weight_decay=args.weight_decay,
    verbose=args.verbose,
    clustering_method=args.clustering,  # "minibatch" or "gpu"
    proxy_frac=args.proxy_frac,
    proxy_cap=args.proxy_cap,
    # Gentle LR + decay for stabilizing local steps
    optimizer_type=args.optimizer_type,  # "sgd", "adam", or "adamw"
    momentum=args.momentum,  # SGD momentum
    use_cosine_decay=args.use_cosine_decay,  # Cosine annealing LR decay
    # Tiny client guard (implementation hygiene)
    min_samples_threshold=args.min_samples_threshold,  # Skip ultra-tiny clients
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

def run_once(label, cfg_overrides=None, num_rounds=args.rounds):
    print(f"\n=== Running: {label} ===")
    coord = make_coordinator(cfg_overrides)
    coord.run(num_rounds=num_rounds)
    #return np.array(coord.validation_curve, dtype=float), np.array(coord.validation_loss_curve, dtype=float)
    return (np.array(coord.validation_curve, dtype=float), 
            np.array(coord.validation_loss_curve, dtype=float),
            np.array(coord.validation_macro_f1_curve, dtype=float))

# ----------------------------
# 4) Define the runs based on --method argument
# ----------------------------
all_runs = {
    "fedsts": ("FedSTS", dict(n_star=None, epsilon=None)),
    "fedstas_no_dp": ("FedSTaS (no-DP)", dict(n_star=BASE_CFG["n_star"], epsilon=None)),
    "fedstas_dp": (f"FedSTaS (ε={args.epsilon})", dict(n_star=BASE_CFG["n_star"], epsilon=args.epsilon)),
}

# Select which method(s) to run
if args.method == "all":
    # Run all available methods
    runs = [all_runs["fedsts"], all_runs["fedstas_no_dp"]]
    if args.epsilon is not None:
        runs.append(all_runs["fedstas_dp"])
else:
    # Run only the specified method
    if args.method == "fedstas_dp" and args.epsilon is None:
        print("ERROR: --epsilon must be provided when running fedstas_dp method")
        sys.exit(1)
    runs = [all_runs[args.method]]

# ----------------------------
# 5) Execute & collect
# ----------------------------
results_acc = {}
results_loss = {}
results_macro_f1 = {}
for label, over in runs:
    acc_curve, loss_curve, macro_f1_curve = run_once(label, over, num_rounds=args.rounds)
    results_acc[label] = acc_curve
    results_loss[label] = loss_curve
    results_macro_f1[label] = macro_f1_curve

# ----------------------------
# 6) Save CSV
# ----------------------------
import pandas as pd
rows = []
for label in results_acc.keys():
    acc_curve = results_acc[label]
    loss_curve = results_loss[label]
    macro_f1_curve = results_macro_f1[label]
    
    # Determine method name for CSV
    if "FedSTS" in label and "FedSTaS" not in label:
        method_name = "FedSTS"
        actual_n_star = 0
        actual_epsilon = None
    elif "no-DP" in label:
        method_name = "FedSTaS_no_DP"
        actual_n_star = args.n_star
        actual_epsilon = None
    else:
        method_name = "FedSTaS_DP"
        actual_n_star = args.n_star
        actual_epsilon = args.epsilon
    
    for r, (acc, loss, macro_f1) in enumerate(zip(acc_curve, loss_curve, macro_f1_curve), start=1):
        rows.append({
            "method": method_name,
            "round": r,
            "test_accuracy": float(acc),
            "test_loss": float(loss),
            "macro_f1": float(macro_f1),
            "beta": args.beta if not args.iid else None,
            "data_distribution": "IID" if args.iid else f"Dirichlet(β={args.beta})",
            "epsilon": actual_epsilon,
            "M": args.M,
            "n_star": actual_n_star,
            "model": args.model,
            "total_clients": args.clients,
            "H": args.h,
            "clients_per_round": args.clients_per_round,
            "local_epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed
        })

df = pd.DataFrame(rows)
df.to_csv(args.csv, index=False)
print(f"\nSaved results to: {os.path.abspath(args.csv)}")
print(f"Total rows: {len(df)}")
print(f"Methods: {df['method'].unique().tolist()}")
print(f"Rounds: {df['round'].min()} to {df['round'].max()}")

'''
%cd /content/fedstas

# Non-DP, β = 0.01 (FedSTS + FedSTaS no-DP)
!python data_cifar10.py --beta 0.01 --rounds 100 --model fast_cnn --clients 100 --h 10 --m_per_round 10 --n_star 2500 --M 300

# With DP (ε = 3), β = 0.01 (adds FedSTaS DP curve)
!python data_cifar10.py --beta 0.01 --epsilon 3 --rounds 100 --model fast_cnn --clients 100 --h 10 --m_per_round 10 --n_star 2500 --M 300
!python main/main_cifar10.py --beta 0.01 --epsilon 3 --rounds 100 --model fast_cnn --clients 100 --h 10 --clients_per_round 10 --n_star 2500 --M 300
!python main/main_cifar10.py --iid --epsilon 3 --rounds 100 --model fast_cnn --clients 100 --h 10 --clients_per_round 10 --n_star 2500 --M 300




import os
os.chdir('/content')

# Clone repository
!git clone https://github.com/JaSlesso/fedstas.git
os.chdir('fedstas')

# Install scikit-learn (required for MiniBatch KMeans)
!pip install -q scikit-learn

# Run with fast stratification (already enabled!)
!python main/main_cifar10.py \
    --iid \
    --epsilon 3 \
    --rounds 50 \
    --model fast_cnn \
    --clients 100 \
    --h 5 \
    --clients_per_round 10 \
    --n_star 2500 \
    --M 300

'''
