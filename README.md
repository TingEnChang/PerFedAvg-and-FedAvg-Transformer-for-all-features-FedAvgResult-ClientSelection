# PerFedAvg / FedAvg Transformer for Consumer Electricity Demand Forecasting

## Overview

A **federated learning** research project for consumer electricity demand forecasting using a **Transformer** model for time-series prediction. The system uses the **Per-FedAvg (Personalized Federated Averaging)** algorithm and supports fast personalization.

### Highlights

- **Privacy**: Client data stays local; only model parameters are shared
- **Personalization**: Per-FedAvg enables fast adaptation of the global model per client
- **Modeling**: Transformer attention for temporal modeling over 25 power-related features
- **Optimization**: Early stopping and learning-rate scheduling to reduce overfitting
- **Visualization**: Tools for performance evaluation and attention visualization
- **Configuration**: YAML-based setup with CUDA / MPS / CPU support
- **Modularity**: Clear separation of components for extension and maintenance

## Quick start

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA or MPS (optional, for GPU acceleration)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/panhyer36/FedAvgTransformerOPSD.git
cd FedAvgTransformerOPSD
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Check configuration**

```bash
python config.py
```

### Training

#### Option 1: Background training (recommended)

```bash
# Start training
./run.sh

# Follow logs
tail -f logs/train_*.log

# Stop training
./stop.sh
```

#### Option 2: Foreground run

```bash
# Train (default: config.yaml)
python train.py

# Train with a custom config
python train.py --config custom_config.yaml

# Test and visualize (default: config.yaml)
python test.py

# Test with a custom config
python test.py --config custom_config.yaml
```

## Project layout

### Core components

```
src/
├── Server.py          #  Federated coordinator (aggregation, global rounds)
├── Client.py          #  Client (local training, validation)
├── Model.py           #  Transformer time-series model
├── DataLoader.py      #  Time-series data loader
└── Trainer.py         #  Trainer (early stopping, LR scheduling)
```

### Data layout

```
data/processed/
├── Consumer_01.csv              # Residential client
├── Consumer_02.csv              # Residential client
├── ...                          # Consumer_03 through Consumer_50
├── Consumer_50.csv              # Residential client
└── Public_Building.csv          # Public building client
```

### Feature description

Each CSV has **25 features** covering appliance usage, weather, and aggregate power:

#### Appliance / aggregate features (15)

```
AC1, AC2, AC3, AC4           # Air conditioning (4 circuits)
Dish washer                  # Dishwasher
Washing Machine              # Washing machine
Dryer                        # Dryer
Water heater                 # Water heater
TV                           # Television
Microwave                    # Microwave
Kettle                       # Electric kettle
Lighting                     # Lighting
Refrigerator                 # Refrigerator
Consumption_Total            # Total consumption
Generation_Total             # Total generation (e.g., solar)
```

#### Weather features (9)

```
TemperatureC                 # Temperature (°C)
DewpointC                    # Dew point (°C)
PressurehPa                  # Pressure (hPa)
WindSpeedKMH                 # Wind speed (km/h)
WindSpeedGustKMH             # Gust speed (km/h)
Humidity                     # Humidity (%)
HourlyPrecipMM               # Hourly precipitation (mm)
dailyrainMM                  # Daily rain (mm)
SolarRadiationWatts_m2       # Solar radiation (W/m²)
```

#### Target (1)

```
Power_Demand                 # Power demand (prediction target)
```

### Client distribution

- **Residential**: 50 clients (`Consumer_01` … `Consumer_50`)
- **Public building**: 1 client (`Public_Building`)
- **Total**: 51 federated clients

### Output directories

```
checkpoints/         #  Model checkpoints
plots/               #  Figures
logs/                #  Training logs
```

## Configuration

### Main file: `config.yaml`

#### Federated learning

```yaml
federated_learning:
  algorithm: "per_fedavg"    # per_fedavg (personalized) or fedavg (standard)
  global_rounds: 100         # Global rounds
  client_fraction: 0.1       # Fraction of clients per round (10%)
  num_clients: 51            # 50 Consumers + 1 Public Building
  eval_interval: 10          # Evaluate every N rounds

  # Per-FedAvg
  per_fedavg:
    meta_step_size: 0.01     # Meta step size (α), personalization speed
    use_second_order: false  # Second-order (HVP): true = HVP, false = first-order
    hvp_damping: 0.01        # HVP damping (HVP mode only)
```

#### Local training

```yaml
local_training:
  local_epochs: 8            # Local epochs per round
  learning_rate: 0.0005
  batch_size: 32
```

#### Model

```yaml
model:
  feature_dim: 25            # Input features (appliances, weather, totals, etc.)
  d_model: 256               # Transformer width
  nhead: 8                   # Attention heads
  num_layers: 4              # Transformer layers
  output_dim: 1              # Output dim (Power_Demand)
  max_seq_length: 100
  dropout: 0.1
```

#### Data

```yaml
data:
  data_path: "data/processed"
  input_length: 96           # Input horizon (96 steps)
  output_length: 1           # Predict one step ahead
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  target: ["Power_Demand"]
```

#### Training / optimization

```yaml
training:
  max_epochs: 200
  early_stopping:
    patience: 15
    min_delta: 0.001
  lr_scheduler:
    patience: 3
    factor: 0.7
    min_lr: 0.000001
```

#### Personalized testing

```yaml
testing:
  personalization_steps: 3   # Per-FedAvg adaptation steps
  support_ratio: 0.2           # Support set ratio for adaptation
  adaptation_lr: 0.001         # Adaptation learning rate
```

#### Device

```yaml
device:
  type: "auto"               # cuda, mps, or cpu (auto picks best)
```

## Usage reference

### Commands

| Command | Purpose | Notes |
|---------|---------|--------|
| `./run.sh` | Background training | `nohup`, logs under `logs/` |
| `./stop.sh` | Stop training | Stops training processes |
| `python train.py` | Foreground training | Default `config.yaml` |
| `python train.py --config my_config.yaml` | Custom config | Training |
| `python test.py` | Test & visualize | Default `config.yaml` |
| `python test.py --config my_config.yaml` | Custom config | Testing |

### Monitoring

```bash
tail -f logs/train_*.log
ps aux | grep python
nvidia-smi   # if using CUDA
```

### Devices

| Type | Platform | Notes |
|------|----------|--------|
| **CUDA** | NVIDIA GPU | Best for large runs |
| **MPS** | Apple Silicon | GPU on Mac |
| **CPU** | Any | Fallback |

Auto selection:

```yaml
device:
  type: "auto"
```

Manual override:

```yaml
device:
  type: "cuda"
  # type: "mps"
  # type: "cpu"
```

## Testing and evaluation

### Two evaluation modes

**Global model**

```bash
python test.py --config config.yaml
```

Evaluates the trained global model on each client’s test split.

**Personalized (Per-FedAvg)**

```bash
python test.py --config config.yaml --personalized
```

Adapts per client before evaluation (deployment-style).

### Metrics

**Regression**

- **MSE**: Mean squared error
- **MAE**: Mean absolute error
- **RMSE**: Root mean squared error
- **R²**: Coefficient of determination (closer to 1 is better)

**Per-FedAvg-oriented**

- Improvement after personalization vs. global-only
- Average gain per adaptation step
- Client heterogeneity in personalization

### Plots (`plots/`)

1. Predictions vs. ground truth (with R²)
2. Time-series prediction comparison
3. Error distribution
4. Attention weights

Per-FedAvg-oriented (when applicable):

5. Before/after personalization
6. Per-client adaptation effects
7. Adaptation loss curves

### Reports

Reports summarize aggregate stats, per-client results, global vs. personalized comparison (when used), and paths to figures.

## Technical notes

### Per-FedAvg workflow

**Per-FedAvg** builds on MAML-style meta-learning in a federated setting.

**Training loop**

1. Server initializes and broadcasts the global model
2. Clients run local Per-FedAvg (first-order or HVP)
3. Server aggregates (federated averaging)
4. Broadcast updated global model
5. Repeat until convergence

**Personalization (evaluation)**

1. Client receives global model
2. Split local data into support and query sets
3. Few gradient steps on the support set
4. Evaluate on the query set

**Per-FedAvg vs. FedAvg**

| Aspect | FedAvg | Per-FedAvg |
|--------|--------|------------|
| Goal | Single global model | Model that personalizes quickly |
| Local update | Standard supervised | Meta-learning toward adaptation |
| Personalization | None | Fast local adaptation |
| Compute | Lower | Moderate |
| Heterogeneity | Limited | Stronger fit |

### Transformer model

- Sinusoidal positional encoding
- 8-head multi-head attention
- 256-d model, 4 encoder layers
- Dropout 0.1
- Sequences up to 100 steps (configurable)

### Data pipeline

1. Load CSV time series (25 features)
2. Time-ordered splits (no leakage)
3. Z-score normalization from training statistics
4. Sliding windows: 96 → 1 step
5. Batched loading

### Personalized testing parameters

```python
personalization_steps: 3
support_ratio: 0.2
adaptation_lr: 0.001
```

Flow: load global model → adapt per client → evaluate on query → compare to global-only.

## Troubleshooting

### Out of memory

```yaml
local_training:
  batch_size: 16  # e.g. reduce from 32
```

### CUDA / MPS errors

```yaml
device:
  type: "cpu"
```

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

### Unstable Per-FedAvg training

```yaml
federated_learning:
  per_fedavg:
    meta_step_size: 0.005
    use_second_order: false
```

### Slow or no convergence

```yaml
local_training:
  learning_rate: 0.0001
  local_epochs: 4
```

### Data loading issues

```bash
python -c "import pandas as pd; pd.read_csv('data/processed/Consumer_01.csv').info()"
```

## Dependencies

### Core

| Package | Version | Role |
|---------|---------|------|
| **torch** | ≥2.0.0 | Deep learning |
| **torchvision** | ≥0.15.0 | Vision utilities (dependency) |
| **numpy** | ≥1.21.0 | Numerics |
| **pandas** | ≥1.3.0 | Data I/O |
| **scikit-learn** | ≥1.0.0 | Metrics / ML utilities |

### Visualization

| Package | Version | Role |
|---------|---------|------|
| **matplotlib** | ≥3.5.0 | Plotting |
| **seaborn** | ≥0.11.0 | Styled stats plots |

### Utilities

| Package | Version | Role |
|---------|---------|------|
| **pyyaml** | ≥6.0 | YAML configs |
| **psutil** | ≥5.8.0 | System monitoring |
| **tensorboard** | ≥2.10.0 | Optional training dashboards |
| **python-dotenv** | — | Environment variables |
| **requests** | — | HTTP |

### Install

```bash
pip install -r requirements.txt
```

Manual minimal set:

```bash
pip install torch>=2.0.0 torchvision>=0.15.0
pip install numpy>=1.21.0 pandas>=1.3.0 scikit-learn>=1.0.0
pip install matplotlib>=3.5.0 seaborn>=0.11.0
pip install pyyaml>=6.0 psutil>=5.8.0
```

### GPU notes

**NVIDIA CUDA**

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Apple Silicon (MPS)**

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```
