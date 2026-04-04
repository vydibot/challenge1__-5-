"""
Atari DQN — Train and Play with Stable-Baselines3
==================================================
This script trains a Deep Q-Network (DQN) agent on any Atari game supported
by the Arcade Learning Environment (ALE) and lets you watch it play.

How it works (high level):
  1. The environment renders raw pixel frames (84 * 84 grayscale after preprocessing).
  2. The agent stacks the last 4 frames to capture motion.
  3. A CNN policy learns which action maximises future reward via the Bellman equation:
       Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',a') − Q(s,a) ]
  4. An epsilon-greedy schedule balances exploration vs exploitation during training.

Usage
-----
  # Train with built-in defaults
  python solaris.py --mode train --model-path models/solaris

  # Train a specific named experiment from the JSON config
  python Solaris.py --mode train --experiment exp_02_lr_high --model-path models/solaris

  # Watch the trained agent play
  python Solaris.py --mode play --model-path models/solaris --episodes 3

  # Run all experiments from sweep_configs.json and keep the best model
  python Solaris.py --mode sweep --sweep-file sweep_configs.json
  #   Each experiment uses the timesteps defined in its JSON entry.
  #   Override all at once with: --timesteps 500000

  # Monitor all sweep runs simultaneously in TensorBoard
  python -m tensorboard.main --logdir logs/solaris/sweep --port 6006

  # Monitor the tuning runs in TensorBoard
  python -m tensorboard.main --logdir logs/tune --port 6006

  # Then open http://localhost:6006

  # Hyperparameter tuning with Optuna
  python Solaris.py --mode tune --n-trials 10 --sampler tpe
  # This runs 10 trials of Optuna optimization using the TPE sampler (Bayesian optimization).
  # The best config is appended to sweep_configs.json so you can easily run it with --mode train or include it in future sweeps.

  # Run replicates of the best config with random seeds
  python Solaris.py --mode replicate --n-replicates 3
  # This runs the best config from sweep_configs.json 3 times with different random seeds,
  # recording each seed in experiment_seeds.json for reproducibility.

Trying a different Atari game
------------------------------
  Change the ENV_ID constant below. Example values:
    "ALE/Pong-v5"
    "ALE/SpaceInvaders-v5"
    "ALE/MsPacman-v5"
    "ALE/Assault-v5"
    "ALE/Asteroids-v5"
  Full list: https://ale.farama.org/environments/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
import datetime

import gc
import torch

import numpy as np
import ale_py
import gymnasium as gym
import optuna

gym.register_envs(ale_py)  # register ALE environments in the gymnasium namespace

from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# CHANGE THIS to try a different Atari game
ENV_ID = "ALE/Solaris-v5"

# Number of consecutive frames stacked together as a single observation.
# Stacking gives the agent a sense of motion (e.g. ball direction/speed).
N_STACK = 4

# Seed tracking storage
SEEDS_DIR = Path("seeds")
SEEDS_FILE = SEEDS_DIR / "experiment_seeds.json"


def set_global_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""

    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def record_seed(experiment_name: str, seed: int, note: str | None = None) -> None:
    """Record used seed(s) for each experiment in seeds/experiment_seeds.json."""

    SEEDS_DIR.mkdir(parents=True, exist_ok=True)

    if SEEDS_FILE.exists():
        with open(SEEDS_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    entry = data.get(experiment_name, {})
    seeds = entry.get("seeds", [])
    if seed not in seeds:
        seeds.append(seed)
    entry["seeds"] = seeds

    if note:
        # Prefer preserving existing note if already present.
        entry.setdefault("note", note)

    data[experiment_name] = entry

    with open(SEEDS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# Logging into Tensor Board

class TensorBoardCallback(BaseCallback):
    """Custom callback that logs per-episode metrics to TensorBoard.

    Attaches to the same SummaryWriter that SB3 creates internally, so our
    custom scalars land in the exact same event file as the built-in
    rollout/ and train/ metrics.  This is why all scalars appear together in
    one TensorBoard run instead of being split across separate directories.

    Scalars added by this callback:
      - training/episode_reward : total reward accumulated in each episode
      - training/epsilon        : current exploration rate (ε), logged every step

    SB3 built-in scalars (also visible in the same run):
      - rollout/ep_rew_mean : rolling mean reward over the last 100 episodes
      - train/loss          : TD-error loss
      - train/learning_rate : current learning rate

    TensorBoard display order (alphabetical by prefix):
      1. rollout/   ← episode reward rolling mean — most useful at a glance
      2. train/     ← loss and learning rate
      3. training/  ← per-episode reward and epsilon from this callback
    """

    def __init__(self) -> None:
        super().__init__()
        self._writer: SummaryWriter | None = None
        self._episode_reward = 0.0

    def _on_training_start(self) -> None:
        # Reuse SB3's own TensorBoard writer so every scalar ends up in the
        # same event file. SB3 stores it inside TensorBoardOutputFormat.writer.
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        for fmt in self.model._logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self._writer = fmt.writer
                return
        # Fallback: SB3 was not given a tensorboard_log dir (should not happen
        # in normal use, but guard against it to avoid AttributeErrors).
        self._writer = None

    def _on_step(self) -> bool:
        if self._writer is None:
            return True

        self._episode_reward += float(self.locals["rewards"][0])

        # Log epsilon every step → smooth decay curve in TensorBoard.
        self._writer.add_scalar("training/epsilon",
                                self.model.exploration_rate,
                                self.num_timesteps)

        if self.locals["dones"][0]:
            self._writer.add_scalar("training/episode_reward",
                                    self._episode_reward,
                                    self.num_timesteps)
            self._episode_reward = 0.0

        return True  # returning False would abort training


# Environment Builders 

def build_training_environment(seed: int) -> VecFrameStack:
    """Create a vectorised, preprocessed Atari environment for training.

    Applies the standard Atari preprocessing pipeline automatically via
    make_atari_env + VecFrameStack:
      - Grayscale conversion
      - Frame resize to 84 * 84
      - Frame skipping (repeat each action 4 steps)
      - Terminal-on-life-loss (treat each life as a separate episode)
      - Frame stacking (last N_STACK frames as one observation)

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A VecFrameStack-wrapped vectorised environment ready for DQN.
    """
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


def build_playing_environment() -> VecFrameStack:
    """Create a human-rendered Atari environment for watching the agent play.

    Differences from the training environment:
      - render_mode="human" opens a visible game window.
      - terminal_on_life_loss=True: on each life loss the wrapper sends the
        FIRE action automatically, so the ball/game restarts without the agent
        getting stuck waiting for input.
      - clip_reward=False: show the real score instead of the clipped {-1,0,+1}.

    Returns:
        A VecFrameStack-wrapped vectorised environment with a human-visible window.
    """
    def _make_single_env() -> AtariWrapper:
        base_env = gym.make(ENV_ID, render_mode="human")
        return AtariWrapper(base_env, terminal_on_life_loss=True, clip_reward=False)

    env = DummyVecEnv([_make_single_env])
    env = VecFrameStack(env, n_stack=N_STACK)
    return env


# Core Logic 

def train_agent(
    model_path: str,
    timesteps: int,
    seed: int,
    tensorboard_log: str,
    hparams: dict | None = None,
) -> float:
    """Train a DQN agent and save the model.

    When called without `hparams` the function uses the built-in defaults
    documented below. The sweep runner passes its own `hparams` dict so the
    same training logic is reused across all experiments.

    Returns:
        Mean episode reward over the last episodes stored in SB3's episode
        info buffer — used by run_sweep to rank experiments.

    Hyperparameter notes (tuned for 8 GB RAM, 300k-step budget):

    All values are scaled proportionally to the 300k step budget.
    DeepMind's original DQN was trained for 50M steps — applying those
    values directly (e.g. target_update=10k, buffer=1M) to a 300k run
    produces only ~28 target syncs and a buffer that takes 333% of the
    budget to fill, both of which cause divergence or wasted capacity.

      learning_rate      1e-4    — standard SB3-Zoo Atari lr; fast enough for
                                    300k steps without overshooting
      buffer_size        50_000  — fills in 50k steps (17% of budget); keeps
                                    experience fresh and RAM usage low
      learning_starts    10_000  — begin learning after 20% buffer fill
      batch_size         64      — stable gradient estimates
      gamma              0.99    — standard Atari discount
      train_freq         4       — one update every 4 steps (DQN paper)
      target_update      1_000   — syncs ~290 times over 300k steps; frequent
                                    enough to track policy improvement without
                                    the instability of 2k (DQN_9 crash at 60k)
      exploration_fraction 0.15  — ε decays over 45k steps, then 255k steps of
                                    pure exploitation; short decay = more useful
                                    gradient signal earlier
      exploration_final_eps 0.01 — 1% random floor; standard Atari value

    Args:
        model_path:      Path (without .zip) where the trained model is saved.
        timesteps:       Total environment steps to train for.
        seed:            Random seed for reproducibility.
        tensorboard_log: Directory where TensorBoard event files are written.
        hparams:         Optional hyperparameter dict; uses built-in defaults
                         when None. Must include all keys listed above plus
                         'env_id', 'timesteps', 'seed'.
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # Ensure reproducible seed for all relevant PRNGs.
    set_global_seed(seed)

    # Use provided hparams or fall back to built-in defaults.
    if hparams is None:
        hparams = dict(
            env_id=ENV_ID,
            learning_rate=1e-4,
            buffer_size=50_000,
            learning_starts=10_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            # WHY 0.15: ε decays over 45k steps, then 255k of pure exploitation.
            exploration_fraction=0.15,
            exploration_final_eps=0.01,
            timesteps=timesteps,
            seed=seed,
        )

    # Write hparams to TensorBoard → visible in the HPARAMS tab.
    # Each run gets its own row so you can compare experiments side-by-side.
    _tb_writer = SummaryWriter(log_dir=tensorboard_log)
    _tb_writer.add_hparams(hparams, metric_dict={"hparam/episode_reward": 0})
    _tb_writer.close()

    env = build_training_environment(seed=seed)

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=hparams["learning_rate"],
        buffer_size=hparams["buffer_size"],
        learning_starts=hparams["learning_starts"],
        batch_size=hparams["batch_size"],
        tau=1.0,
        gamma=hparams["gamma"],
        train_freq=hparams["train_freq"],
        gradient_steps=1,
        target_update_interval=hparams["target_update_interval"],
        exploration_fraction=hparams["exploration_fraction"],
        exploration_final_eps=hparams["exploration_final_eps"],
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=seed,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=TensorBoardCallback(),
        progress_bar=True,
    )
    model.save(model_path)
    env.close()
    print(f"Model saved → {model_path}.zip")

    # Return mean episode reward over the last recorded episodes.
    # SB3 maintains ep_info_buffer (deque of {r, l, t} dicts) during training.
    if model.ep_info_buffer:
        return float(np.mean([ep["r"] for ep in model.ep_info_buffer]))
    return 0.0


def play_agent(model_path: str, episodes: int) -> None:
    """Load a trained model and watch it play in a visible game window.

    Each full game (all lives exhausted) counts as one episode. The agent
    automatically fires at the start of each life thanks to the FireResetEnv
    wrapper inside build_playing_environment().

    Args:
        model_path: Path to the saved model (with or without .zip extension).
        episodes:   Number of full games to play before exiting.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(
            f"Model not found: {model_path}.zip\n"
            "Run with --mode train first to create a model."
        )

    env = build_playing_environment()
    model = DQN.load(model_path, env=env)

    completed = 0
    obs = env.reset()
    episode_reward = 0.0

    while completed < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        episode_reward += float(rewards[0])

        if dones[0]:
            # 'lives' > 0 means a mid-game life loss; the env auto-resets and fires.
            # 'lives' == 0 means the full game is over — count it as one episode.
            if infos[0].get("lives", 0) == 0:
                completed += 1
                print(f"Episode {completed}/{episodes}  reward: {episode_reward:.2f}")
                episode_reward = 0.0

    env.close()


def run_replicates(
    sweep_path: str,
    n_replicates: int,
    base_log_dir: str,
    model_base_path: str,
) -> None:
    """Run the best config from sweep_configs.json multiple times with different random seeds.

    This is useful for validating the stability of the best hyperparameters found by Optuna.
    Each replicate uses a random seed and records it in experiment_seeds.json.

    Args:
        sweep_path:      Path to the JSON file containing experiment configs.
        n_replicates:    Number of times to run the best config.
        base_log_dir:    Root TensorBoard log directory.
        model_base_path: Base path for saving replicate models (without .zip).
    """
    with open(sweep_path) as f:
        configs = json.load(f)

    if not configs:
        raise ValueError(f"No configs found in {sweep_path}")

    # Assume the last config is the best (from Optuna)
    best_config = configs[-1]
    best_name = best_config["name"]

    print(f"Running {n_replicates} replicates of best config: {best_name}")
    print(f"  {best_config.get('note', '')}")
    print(f"{'='*60}")

    import random
    for i in range(1, n_replicates + 1):
        # Generate a random seed
        replicate_seed = random.randint(0, 100000)

        experiment_name = f"{best_name}_replicate_{i}"

        print(f"Replicate {i}/{n_replicates}: {experiment_name} (seed={replicate_seed})")

        # Build hparams dict
        hparams = {
            "env_id":                 ENV_ID,
            "learning_rate":          best_config["learning_rate"],
            "buffer_size":            best_config["buffer_size"],
            "learning_starts":        best_config["learning_starts"],
            "batch_size":             best_config["batch_size"],
            "gamma":                  best_config["gamma"],
            "train_freq":             best_config["train_freq"],
            "target_update_interval": best_config["target_update_interval"],
            "exploration_fraction":   best_config["exploration_fraction"],
            "exploration_final_eps":  best_config["exploration_final_eps"],
            "timesteps":              best_config["timesteps"],
            "seed":                   replicate_seed,
        }

        model_path = f"{model_base_path}_replicate_{i}"
        log_dir = f"{base_log_dir}/replicates/{experiment_name}"

        record_seed(experiment_name, replicate_seed, note=best_config.get("note", ""))

        print(f"  Seed {replicate_seed} recorded for {experiment_name}")

        score = train_agent(
            model_path=model_path,
            timesteps=best_config["timesteps"],
            seed=replicate_seed,
            tensorboard_log=log_dir,
            hparams=hparams,
        )
        print(f"  → final mean reward: {score:.2f}")

    print(f"\n{'='*60}")
    print(f"Replication complete. Models saved as {model_base_path}_replicate_1.zip etc.")
    print(f"TensorBoard logs: {base_log_dir}/replicates/")


# Sweep 

def run_sweep(
    sweep_path: str,
    default_timesteps: int,
    seed: int,
    base_log_dir: str,
    best_model_path: str,
) -> None:
    """Run all experiments defined in a JSON config file and save the best model.

    Each experiment uses the ``timesteps`` value from its JSON entry. If the
    entry omits ``timesteps``, the value from ``--timesteps`` (default 300k)
    is used as a fallback. This lets you give expensive experiments a larger
    budget and quick exploratory ones a smaller one, all in one sweep.

    TensorBoard logs for every run are written to
    ``<base_log_dir>/sweep/<experiment_name>/`` so all runs are visible
    together by pointing TensorBoard at ``<base_log_dir>/sweep``.

    After all experiments finish, only the model with the highest mean final
    episode reward is kept at ``best_model_path``. All intermediate models are
    deleted to save disk space.

    Args:
        sweep_path:        Path to the JSON file containing experiment configs.
        default_timesteps: Fallback timestep budget for experiments that do not
                           define ``timesteps`` in their JSON entry.
        seed:              Random seed applied to every experiment.
        base_log_dir:      Root TensorBoard log directory.
        best_model_path:   Where to save the winning model (without .zip).
    """
    with open(sweep_path) as f:
        configs = json.load(f)

    tmp_model_dir = Path("models") / "_sweep_tmp"
    tmp_model_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, float]] = []
    total = len(configs)

    for idx, cfg in enumerate(configs, start=1):
        name = cfg.get("name", f"exp_{idx:02d}")
        note = cfg.get("note", "")
        # Per-experiment timestep budget; falls back to the CLI --timesteps value.
        exp_timesteps = cfg.get("timesteps", default_timesteps)

        # Ensure each experiment gets a reproducible but different seed.
        experiment_seed = seed + idx

        print(f"Experiment {idx}/{total}: {name}  ({exp_timesteps:,} steps, seed={experiment_seed})")
        if note:
            print(f"  {note}")
        print(f"{'='*60}")

        # Build the hparams dict expected by train_agent.
        hparams = {
            "env_id":                 ENV_ID,
            "learning_rate":          cfg["learning_rate"],
            "buffer_size":            cfg["buffer_size"],
            "learning_starts":        cfg["learning_starts"],
            "batch_size":             cfg["batch_size"],
            "gamma":                  cfg["gamma"],
            "train_freq":             cfg["train_freq"],
            "target_update_interval": cfg["target_update_interval"],
            "exploration_fraction":   cfg["exploration_fraction"],
            "exploration_final_eps":  cfg["exploration_final_eps"],
            "timesteps":              exp_timesteps,
            "seed":                   experiment_seed,
        }

        model_path = str(tmp_model_dir / name)
        log_dir   = f"{base_log_dir}/sweep/{name}"

        record_seed(name, experiment_seed, note=note)

        score = train_agent(
            model_path=model_path,
            timesteps=exp_timesteps,
            seed=experiment_seed,
            tensorboard_log=log_dir,
            hparams=hparams,
        )
        results.append((name, score))
        print(f"  → final mean reward: {score:.2f}")

    # Summary
    results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = results[0]

    print(f"\n{'='*60}")
    print("Sweep complete — results ranked by final mean reward:")
    for rank, (name, score) in enumerate(results, start=1):
        marker = "  BEST" if rank == 1 else ""
        print(f"  {rank:2d}. {name:<35s}  {score:7.2f}{marker}")
    print(f"{'='*60}")

    # Save best, clean up 
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(tmp_model_dir / f"{best_name}.zip"), f"{best_model_path}.zip")
    shutil.rmtree(tmp_model_dir)

    print(f"\nBest model ({best_name}, score={best_score:.2f}) saved into {best_model_path}.zip")
    print(f"TensorBoard logs for all runs: {base_log_dir}/sweep/")


def inspect_model(model_path: str) -> None:
    """Load a saved model and print its hyperparameters.

    SB3 serialises all constructor arguments inside the .zip file, so this
    works for any model saved by this script — including runs made before
    hyperparameter logging was added to TensorBoard.

    Args:
        model_path: Path to the saved model (with or without .zip extension).

    Example:
        python solaris.py --mode inspect --model-path models/solaris
    """
    if not os.path.exists(f"{model_path}.zip"):
        raise FileNotFoundError(f"Model not found: {model_path}.zip")

    model = DQN.load(model_path)

    # Parameters SB3 saves inside the zip
    params = {
        "policy":                  model.policy_class.__name__,
        "learning_rate":           model.learning_rate,
        "buffer_size":             model.buffer_size,
        "learning_starts":         model.learning_starts,
        "batch_size":              model.batch_size,
        "tau":                     model.tau,
        "gamma":                   model.gamma,
        "train_freq":              model.train_freq,
        "gradient_steps":          model.gradient_steps,
        "target_update_interval":  model.target_update_interval,
        "exploration_fraction":    model.exploration_fraction,
        "exploration_final_eps":   model.exploration_final_eps,
        "num_timesteps_trained":   model.num_timesteps,
    }

    print(f"\n── Saved model: {model_path}.zip")
    for key, value in params.items():
        print(f"  {key:30s}: {value}")
    print("─" * 55 + "\n")


# CLI 

def parse_args() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train or watch a DQN agent on an Atari game.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["train", "play", "inspect", "sweep", "tune", "replicate"], required=True,
        help="'train' single run, 'play' watch agent, 'inspect' print params, "
             "'sweep' run all experiments from --sweep-file, 'tune' run Optuna optimization, "
             "'replicate' run best config from --sweep-file multiple times with random seeds.",
    )
    parser.add_argument(
        "--sweep-file", default="sweep_configs.json",
        help="Path to JSON file with experiment configs (used by --mode sweep and --experiment).",
    )
    parser.add_argument(
        "--experiment", default=None,
        help="Name of a single experiment in --sweep-file to run with --mode train. "
             "Uses built-in defaults when omitted.",
    )
    parser.add_argument(
        "--model-path", default="models/solaris",
        help="Path to save (train) or load (play) the model (without .zip).",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training steps. Overrides the value in the JSON config when set. "
             "Defaults to the JSON value, or 300k if neither is specified.",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of full games to play in play mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tensorboard-log", default="logs/solaris",
        help="Directory for TensorBoard logs.",
    )
    parser.add_argument(
        "--n-replicates", type=int, default=3,
        help="Number of replicates to run in replicate mode.",
    )
    parser.add_argument(
        "--sampler", choices=["tpe", "random"], default="tpe",
        help="Optuna sampler for tuning: 'tpe' for TPESampler, 'random' for RandomSampler.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "train":
        hparams = None
        timesteps = args.timesteps or 300_000

        if args.experiment:
            # Load a named experiment from the JSON so the same configs work
            # for both single runs (--mode train) and full sweeps (--mode sweep).
            with open(args.sweep_file) as f:
                configs = {c["name"]: c for c in json.load(f)}
            if args.experiment not in configs:
                raise ValueError(
                    f"Experiment '{args.experiment}' not found in {args.sweep_file}.\n"
                    f"Available: {', '.join(configs)}"
                )
            cfg = configs[args.experiment]
            timesteps = args.timesteps or cfg.get("timesteps", 300_000)
            hparams = {
                "env_id":                 ENV_ID,
                "learning_rate":          cfg["learning_rate"],
                "buffer_size":            cfg["buffer_size"],
                "learning_starts":        cfg["learning_starts"],
                "batch_size":             cfg["batch_size"],
                "gamma":                  cfg["gamma"],
                "train_freq":             cfg["train_freq"],
                "target_update_interval": cfg["target_update_interval"],
                "exploration_fraction":   cfg["exploration_fraction"],
                "exploration_final_eps":  cfg["exploration_final_eps"],
                "timesteps":              timesteps,
                "seed":                   args.seed,
            }
            print(f"Loaded experiment '{args.experiment}' from {args.sweep_file}")
            record_seed(args.experiment, args.seed, note=cfg.get("note", ""))
        else:
            record_seed("manual", args.seed, note="manual train run")

        train_agent(
            model_path=args.model_path,
            timesteps=timesteps,
            seed=args.seed,
            tensorboard_log=args.tensorboard_log,
            hparams=hparams,
        )

    elif args.mode == "play":
        play_agent(model_path=args.model_path, episodes=args.episodes)

    elif args.mode == "sweep":
        run_sweep(
            sweep_path=args.sweep_file,
            default_timesteps=args.timesteps or 300_000,
            seed=args.seed,
            base_log_dir=args.tensorboard_log,
            best_model_path=args.model_path,
        )

    elif args.mode == "tune":
        if args.sampler == "tpe":
            sampler = optuna.samplers.TPESampler()
        elif args.sampler == "random":
            sampler = optuna.samplers.RandomSampler()
        else:
            raise ValueError(f"Unknown sampler: {args.sampler}")

        tuner = SolarisHyperparameterTuner(sampler=sampler)
        print(f"Starting Optuna optimization with {args.sampler} sampler for {args.n_trials} trials...")
        tuner.optimize(n_trials=args.n_trials)
        tuner.save_to_sweep_config(filepath=args.sweep_file)
        print(f"Optimization complete. Best config appended to {args.sweep_file}")


    elif args.mode == "replicate":
        run_replicates(
            sweep_path=args.sweep_file,
            n_replicates=args.n_replicates,
            base_log_dir=args.tensorboard_log,
            model_base_path=args.model_path,
        )

    else:
        inspect_model(model_path=args.model_path)


class SolarisHyperparameterTuner:
    """Memory-optimized Hyperparameter tuner for Solaris using Optuna."""

    def __init__(self, sampler=None):
        # Default to TPESampler (Bayesian Optimization)
        self.sampler = sampler or optuna.samplers.TPESampler()
        # MedianPruner: Kills trials that perform worse than the median of previous runs
        self.pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        self.study = None
        self.best_trial = None

    def _objective(self, trial):
        """Objective function with memory safeguards."""
        
        learning_rate = trial.suggest_float("learning_rate",1e-5, 5e-4, log=True)
        
        # Buffer capped at 50k to prevent OOM on 16GB RAM
        buffer_size = trial.suggest_categorical("buffer_size", [10000, 20000, 50000])
        
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        
        learning_starts = trial.suggest_int("learning_starts", 1000, 20000)
        gamma = trial.suggest_categorical("gamma", [0.95, 0.99, 0.995])
        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8])
        target_update_interval = trial.suggest_int("target_update_interval", 500, 5000)
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.3)
        exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.05)

        hparams = {
            "env_id": ENV_ID,
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "learning_starts": learning_starts,
            "batch_size": batch_size,
            "gamma": gamma,
            "train_freq": train_freq,
            "target_update_interval": target_update_interval,
            "exploration_fraction": exploration_fraction,
            "exploration_final_eps": exploration_final_eps,
            "timesteps": 300000,
            "seed": 42 + trial.number, # Vary seed slightly per trial
        }

        # Paths
        trial_name = f"trial_{trial.number}"
        model_path = f"models/tune/{trial_name}"
        tensorboard_log = f"logs/tune/{trial_name}"

        # Format hyperparameters for note
        hparams_note = (
            f"lr={learning_rate:.2e}, buffer={buffer_size}, batch={batch_size}, "
            f"starts={learning_starts}, gamma={gamma}, train_freq={train_freq}, "
            f"target_update={target_update_interval}, exp_frac={exploration_fraction:.2f}, "
            f"exp_eps={exploration_final_eps:.2f}"
        )

        # Record seed with hyperparameter summary
        record_seed(trial_name, 42 + trial.number, note=hparams_note)

        # --- 2. EXECUTION WITH CLEANUP ---
        reward = 0.0
        try:
            reward = train_agent(
                model_path=model_path,
                timesteps=300000,
                seed=42 + trial.number,
                tensorboard_log=tensorboard_log,
                hparams=hparams,
            )
        except Exception as e:
            print(f"Trial {trial.number} failed due to: {e}")
            return -9999.0 
        finally:
            gc.collect()
            torch.cuda.empty_cache()

        return reward

    def optimize(self, n_trials=30):
        """Run optimization with pruning enabled."""
        self.study = optuna.create_study(
            sampler=self.sampler, 
            pruner=self.pruner, 
            direction="maximize"
        )
        self.study.optimize(self._objective, n_trials=n_trials, n_jobs=1)
        self.best_trial = self.study.best_trial

    def get_best_config(self):
        """Formats the result for sweep_configs.json."""
        if self.best_trial is None:
            raise ValueError("No best trial found.")

        # Get simplified sampler name (e.g. 'tpe' or 'random')
        sampler_type = self.sampler.__class__.__name__.replace("Sampler", "").lower()
        
        # Format hyperparameters in note
        hparams_note = (
            f"lr={self.best_trial.params['learning_rate']:.2e}, "
            f"buffer={self.best_trial.params['buffer_size']}, "
            f"batch={self.best_trial.params['batch_size']}, "
            f"starts={self.best_trial.params['learning_starts']}, "
            f"gamma={self.best_trial.params['gamma']}, "
            f"train_freq={self.best_trial.params['train_freq']}, "
            f"target_update={self.best_trial.params['target_update_interval']}, "
            f"exp_frac={self.best_trial.params['exploration_fraction']:.2f}, "
            f"exp_eps={self.best_trial.params['exploration_final_eps']:.2f}"
        )
        
        return {
            "name": f"optuna_{sampler_type}_best",
            "note": f"Auto-tuned Solaris via Optuna {sampler_type} | {hparams_note}",
            "timesteps": 300000,
            "learning_rate": self.best_trial.params["learning_rate"],
            "buffer_size": self.best_trial.params["buffer_size"],
            "learning_starts": self.best_trial.params["learning_starts"],
            "batch_size": self.best_trial.params["batch_size"],
            "gamma": self.best_trial.params["gamma"],
            "train_freq": self.best_trial.params["train_freq"],
            "target_update_interval": self.best_trial.params["target_update_interval"],
            "exploration_fraction": self.best_trial.params["exploration_fraction"],
            "exploration_final_eps": self.best_trial.params["exploration_final_eps"],
        }

    def save_to_sweep_config(self, filepath="sweep_configs.json"):
        config = self.get_best_config()
        path = Path(filepath)

        data = []
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                elif not isinstance(data, list):
                    raise ValueError(
                        f"{filepath} must contain a JSON list or object, got "
                        f"{type(data).__name__} instead."
                    )
            except json.JSONDecodeError:
                backup_path = path.with_suffix(path.suffix + ".backup")
                shutil.copy(path, backup_path)
                print(
                    f"Warning: failed to parse {filepath}; backed up original to "
                    f"{backup_path}. Old data will be replaced by the new best config."
                )
                data = []

        if any(isinstance(entry, dict) and entry.get("name") == config["name"] for entry in data):
            print(f"Sweep config already contains entry {config['name']}; skipping duplicate append.")
        else:
            data.append(config)

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Successfully added best trial to {filepath}")


if __name__ == "__main__":
    main() 