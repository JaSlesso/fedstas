# Google Colab Experiment Commands

## Setup Instructions

### 1. Install cuML for GPU KMeans (Fast Clustering)

```bash
# Install cuML for CUDA 11 (most Colab instances)
!pip install cuml-cu11 --quiet

# OR if you have CUDA 12:
# !pip install cuml-cu12 --quiet

# Install other dependencies (if needed)
!pip install scikit-learn pandas matplotlib --quiet
```

### 2. Clone Repository (if needed)

```bash
# Navigate to workspace
import os
os.chdir('/content')

# Clone your repository
!git clone https://github.com/YOUR_USERNAME/fedstas.git
os.chdir('fedstas')

# Verify files
!ls -la main/
```

---

## Experiment Overview

**Total Runs**: 9 commands (3 setups × 3 methods)

**Fixed Parameters** (always default, not specified):
- `M=300`
- `model=fast_cnn`
- `clients=100`
- `h=10`
- `clients_per_round=10`
- `proxy_frac=0.30`
- `proxy_cap=128`
- `d_prime=5`
- `restratify_every=20`
- `clustering=gpu` (default now)
- `verbose=True` (default now)

**Varying Parameters**:
- Data distribution: IID, beta=0.01, beta=0.1
- Methods: FedSTS, FedSTaS with DP (ε=3), FedSTaS no DP
- Other training params: rounds, n_star, epsilon, batch_size, lr, epochs, weight_decay, seed

---

## Setup 1: IID Distribution (3 commands)

### Command 1.1: IID - FedSTS (baseline, no data sampling, no DP)
```bash
!python main/main_cifar10.py \
  --iid \
  --rounds 300 \
  --n_star 0 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method fedsts \
  --csv results_iid_fedsts.csv
```

### Command 1.2: IID - FedSTaS with DP (ε=3)
```bash
!python main/main_cifar10.py \
  --iid \
  --epsilon 3.0 \
  --rounds 300 \
  --n_star 2500 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method fedstas_dp \
  --csv results_iid_fedstas_dp.csv
```

### Command 1.3: IID - FedSTaS without DP
```bash
!python main/main_cifar10.py \
  --iid \
  --rounds 300 \
  --n_star 2500 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method fedstas_no_dp \
  --csv results_iid_fedstas_nodp.csv
```

---

## Setup 2: Strong Non-IID (beta=0.01) - 3 commands

### Command 2.1: Beta=0.01 - FedSTS (baseline)
```bash
!python main/main_cifar10.py \
  --beta 0.01 \
  --rounds 300 \
  --n_star 0 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method fedsts \
  --csv results_beta0.01_fedsts.csv
```

### Command 2.2: Beta=0.01 - FedSTaS with DP (ε=3)
```bash
!python main/main_cifar10.py \
  --beta 0.01 \
  --epsilon 3.0 \
  --rounds 300 \
  --n_star 2500 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method fedstas_dp \
  --csv results_beta0.01_fedstas_dp.csv
```

### Command 2.3: Beta=0.01 - FedSTaS without DP
```bash
!python main/main_cifar10.py \
  --beta 0.01 \
  --rounds 300 \
  --n_star 2500 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method fedstas_no_dp \
  --csv results_beta0.01_fedstas_nodp.csv
```

---

## Setup 3: Moderate Non-IID (beta=0.1) - 3 commands

### Command 3.1: Beta=0.1 - FedSTS (baseline)
```bash
!python main/main_cifar10.py \
  --beta 0.1 \
  --rounds 300 \
  --n_star 0 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method fedsts \
  --csv results_beta0.1_fedsts.csv
```

### Command 3.2: Beta=0.1 - FedSTaS with DP (ε=3)
```bash
!python main/main_cifar10.py \
  --beta 0.1 \
  --epsilon 3.0 \
  --rounds 300 \
  --n_star 2500 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method fedstas_dp \
  --csv results_beta0.1_fedstas_dp.csv
```

### Command 3.3: Beta=0.1 - FedSTaS without DP
```bash
!python main/main_cifar10.py \
  --beta 0.1 \
  --rounds 300 \
  --n_star 2500 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method fedstas_no_dp \
  --csv results_beta0.1_fedstas_nodp.csv
```

---

## Generate Plots for All Results

### Plot Setup 1 (IID)
```bash
# You'll need to combine the 3 CSVs or run all methods at once
# Alternative: Run with --method all to get all three methods in one CSV
!python main/main_cifar10.py \
  --iid \
  --epsilon 3.0 \
  --rounds 300 \
  --n_star 2500 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method all \
  --csv results_iid_all.csv

# Then plot
!python main_plot.py --csv results_iid_all.csv --output plots/iid_comparison.png
```

### Plot Setup 2 (Beta=0.01)
```bash
!python main/main_cifar10.py \
  --beta 0.01 \
  --epsilon 3.0 \
  --rounds 300 \
  --n_star 2500 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method all \
  --csv results_beta0.01_all.csv

!python main_plot.py --csv results_beta0.01_all.csv --output plots/beta0.01_comparison.png
```

### Plot Setup 3 (Beta=0.1)
```bash
!python main/main_cifar10.py \
  --beta 0.1 \
  --epsilon 3.0 \
  --rounds 300 \
  --n_star 2500 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method all \
  --csv results_beta0.1_all.csv

!python main_plot.py --csv results_beta0.1_all.csv --output plots/beta0.1_comparison.png
```

---

## Alternative: Run All Methods at Once (Recommended!)

Instead of 9 separate commands, you can run just 3 commands (one per setup):

### Simplified Command Set (3 total)

```bash
# Setup 1: IID - All methods
!python main/main_cifar10.py --iid --epsilon 3.0 --rounds 300 --n_star 2500 --epochs 2 --batch_size 64 --lr 0.01 --weight_decay 1e-5 --seed 562 --method all --csv results_iid_all.csv

# Setup 2: Beta=0.01 - All methods
!python main/main_cifar10.py --beta 0.01 --epsilon 3.0 --rounds 300 --n_star 2500 --epochs 2 --batch_size 64 --lr 0.01 --weight_decay 1e-5 --seed 562 --method all --csv results_beta0.01_all.csv

# Setup 3: Beta=0.1 - All methods
!python main/main_cifar10.py --beta 0.1 --epsilon 3.0 --rounds 300 --n_star 2500 --epochs 2 --batch_size 64 --lr 0.01 --weight_decay 1e-5 --seed 562 --method all --csv results_beta0.1_all.csv
```

**Benefits**:
- 3 commands instead of 9
- All methods compared in one run
- Consistent global model initialization
- Easier to plot comparisons

---

## Download Results from Colab

```python
from google.colab import files

# Download CSV results
files.download('results_iid_all.csv')
files.download('results_beta0.01_all.csv')
files.download('results_beta0.1_all.csv')

# Download plots
files.download('plots/iid_comparison.png')
files.download('plots/beta0.01_comparison.png')
files.download('plots/beta0.1_comparison.png')
```

---

## Expected Runtime

Per setup (300 rounds, 100 clients, 10 clients/round):
- **FedSTS**: ~30-40 minutes
- **FedSTaS no DP**: ~30-40 minutes
- **FedSTaS with DP**: ~35-45 minutes

**Total time**:
- 9 separate commands: ~5-6 hours
- 3 combined commands (--method all): ~2-3 hours

**Recommendation**: Use `--method all` for efficiency!

---

## Monitor Progress

Check GPU usage:
```bash
!nvidia-smi
```

Check current progress (in another cell while running):
```bash
!tail -f cifar10_beta_eps_results.csv
```

Or view logs:
```python
# View recent output
!tail -n 50 your_output.log
```

---

## Troubleshooting

### If cuML fails to install:
```bash
# Fallback to CPU clustering
# Change --clustering gpu to --clustering minibatch
!python main/main_cifar10.py --clustering minibatch [other args...]
```

### If out of memory:
```bash
# Reduce batch size or clients per round
!python main/main_cifar10.py --batch_size 32 --clients_per_round 5 [other args...]
```

### If disconnected from Colab:
```python
# Save checkpoints periodically (need to implement)
# Or use shorter --rounds for testing (e.g., --rounds 50)
```

---

## Complete Colab Notebook Template

```python
# Cell 1: Setup
!pip install cuml-cu11 --quiet
!pip install scikit-learn pandas matplotlib --quiet

import os
os.chdir('/content')

# Clone repository (replace with your repo)
!git clone https://github.com/YOUR_USERNAME/fedstas.git
os.chdir('fedstas')

# Cell 2: Verify GPU
!nvidia-smi

# Cell 3: Run Setup 1 (IID)
!python main/main_cifar10.py \
  --iid \
  --epsilon 3.0 \
  --rounds 300 \
  --n_star 2500 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method all \
  --csv results_iid_all.csv

# Cell 4: Generate Plot for Setup 1
!python main_plot.py --csv results_iid_all.csv --output plots/iid_comparison.png

# Cell 5: Run Setup 2 (Beta=0.01)
!python main/main_cifar10.py \
  --beta 0.01 \
  --epsilon 3.0 \
  --rounds 300 \
  --n_star 2500 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method all \
  --csv results_beta0.01_all.csv

# Cell 6: Generate Plot for Setup 2
!python main_plot.py --csv results_beta0.01_all.csv --output plots/beta0.01_comparison.png

# Cell 7: Run Setup 3 (Beta=0.1)
!python main/main_cifar10.py \
  --beta 0.1 \
  --epsilon 3.0 \
  --rounds 300 \
  --n_star 2500 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.01 \
  --weight_decay 1e-5 \
  --seed 562 \
  --method all \
  --csv results_beta0.1_all.csv

# Cell 8: Generate Plot for Setup 3
!python main_plot.py --csv results_beta0.1_all.csv --output plots/beta0.1_comparison.png

# Cell 9: Download Results
from google.colab import files
files.download('results_iid_all.csv')
files.download('results_beta0.01_all.csv')
files.download('results_beta0.1_all.csv')
files.download('plots/iid_comparison.png')
files.download('plots/beta0.01_comparison.png')
files.download('plots/beta0.1_comparison.png')
```

---

## Summary

**Total Experiments**: 3 setups × 3 methods = 9 runs

**Recommended Approach**: Use `--method all` (3 commands instead of 9)

**Fixed Parameters** (using defaults):
- M=300, model=fast_cnn, clients=100, h=10, clients_per_round=10
- proxy_frac=0.30, proxy_cap=128, d_prime=5, restratify_every=20
- clustering=gpu (default), verbose=True (default)

**Key Differences Between Setups**:
1. **IID**: Uniform data distribution (easier optimization)
2. **Beta=0.01**: Strong non-IID (realistic federated setting)
3. **Beta=0.1**: Moderate non-IID (between IID and beta=0.01)

**Key Differences Between Methods**:
1. **FedSTS**: No data sampling (n_star=0), no DP
2. **FedSTaS no DP**: Data sampling (n_star=2500), no DP
3. **FedSTaS with DP**: Data sampling (n_star=2500), DP with ε=3

Good luck with your experiments! 🚀

