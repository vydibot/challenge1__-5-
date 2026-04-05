# Solaris Atari DQN

This repository contains a Reinforcement Learning project for training and evaluating a DQN agent on Atari games using Stable-Baselines3.

## Repository structure

- `Solaris.py` — main application entry point for training, playing, inspecting, tuning, sweeping, and replicating experiments.
- `sweep_configs.json` — experiment definitions used by sweep and replicate commands.
- `variance_analysis/` — notebook and data for analyzing reward/length variance between empirical and Optuna runs.
- `models/` — saved model artifacts.
- `logs/` — TensorBoard logs.
- `seeds/experiment_seeds.json` — recorded random seeds for reproducibility.
- `Dockerfile` — container build configuration.
- `docker-compose.yml` — simple compose setup for running the app inside Docker.

## Setup

### Option 1: Local virtual environment

```bash
cd ./challenge1__5
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install poetry
poetry install
```

### Option 2: Docker

Build the Docker image:

```bash
docker build -t challenge1 .
```

Run an interactive container:

```bash
docker run --rm -it -v "$PWD":/app -p 8888:8888 challenge1 bash
```

Or use Docker Compose:

```bash
docker compose up --build
```

If you want to run Jupyter Lab from inside the container:

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## Using Solaris.py

### Common flags

- `--mode` — operation mode. One of:
  - `train`
  - `play`
  - `inspect`
  - `sweep`
  - `tune`
  - `replicate`
- `--model-path` — model save/load base path (without `.zip`). Default: `models/solaris`
- `--sweep-file` — config file for `sweep`, `experiment`, and `replicate`. Default: `sweep_configs.json`
- `--experiment` — a named experiment from `--sweep-file` to run in `train` mode.
- `--timesteps` — override training steps from config. Default fallback: `300000`.
- `--seed` — random seed for reproducibility.
- `--tensorboard-log` — log directory for TensorBoard. Default: `logs/solaris`

### Train a model

```bash
python Solaris.py --mode train --model-path models/solaris
```

Train a specific experiment from `sweep_configs.json`:

```bash
python Solaris.py --mode train --experiment exp_02_lr_high --model-path models/solaris
```

Override the timestep budget:

```bash
python Solaris.py --mode train --timesteps 500000 --model-path models/solaris
```

### Play a saved agent

```bash
python Solaris.py --mode play --model-path models/solaris --episodes 3
```

### Sweep experiments

Run all experiments defined in `sweep_configs.json` and keep the best model:

```bash
python Solaris.py --mode sweep --sweep-file sweep_configs.json --model-path models/solaris
```

### Tune with Optuna

Run hyperparameter optimization with the TPE sampler:

```bash
python Solaris.py --mode tune --n-trials 10 --sampler tpe
```

Run with random search instead:

```bash
python Solaris.py --mode tune --n-trials 10 --sampler random
```

The best configuration is appended to `sweep_configs.json`.

### Replicate best config

Run the best config from `sweep_configs.json` multiple times with different random seeds:

```bash
python Solaris.py --mode replicate --n-replicates 3
```

This records reproducible seeds in `seeds/experiment_seeds.json`.

### Inspect a saved model

```bash
python Solaris.py --mode inspect --model-path models/solaris
```

## TensorBoard

To visualize training logs(Empirical):

```bash
python -m tensorboard.main --logdir logs/solaris/sweep --port 6006
```

To visualize training logs(Optuna):

```bash
python -m tensorboard.main --logdir logs/tune --port 6006
```

To visualize training logs(Best of optuna and Empirical with different random seeds):

```bash
python -m tensorboard.main --logdir logs/solaris/replicates --port 6006
```


Then open `http://localhost:6006`.

## Notes for reproducibility

- The Dockerfile installs dependencies via Poetry and mounts the repo into `/app`.
- The `docker-compose.yml` file provides a simple local container service.
- Use the `--seed` option to make training runs reproducible.
- Seeds are recorded automatically by `Solaris.py` when training with `--mode train` or `--mode replicate`.

## Recommended workflow

1. Build the Docker image or set up the local virtualenv.
2. Run training with `--mode train`.
3. Evaluate with `--mode play`.
4. Use `--mode tune` to optimize hyperparameters.
5. Validate the best results with `--mode replicate`.

## Video

### Video description

This video presents the Solaris Atari DQN project and walks through the agent design, hardware limits, hyperparameter strategy, empirical vs Optuna tuning, TensorBoard results, and final gameplay demonstrations.

### Time marks

- `0:40` — Description of agent and general characteristics
- `1:30` — Explanation of hardware limitations
- `1:50` — Explanation of hyperparameter selection strategies
- `3:12` — TensorBoard results for empirical (manual hyperparameter selection)
- `3:40` — Explanation of Bayesian optimization using Optuna (TPE)
- `4:30` — Results using Optuna and the best result (TensorBoard results)
- `5:45` — Random-seeds experiments for the best result on both approaches
- `6:48` — Variance analysis of results for best hyperparameters: Optuna vs empirical
- `9:00` — Final model gameplay demonstration (Optuna)

### Link

Add the video link here: `https://drive.google.com/file/d/1AiRy0wg7LL1oPYL17EXwwgzqK_-oUD2d/view?usp=sharing`

