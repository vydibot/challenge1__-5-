import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
    AtariPreprocessing,
)

def make_env(env_id: str, seed: int = 0):
    """Build a pre-processed \texttt{ALE} environment compatible with \
    \texttt{PPO}."""
    env = gym.make(env_id, render_mode=None)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=True,        # pixel values in [0, 1]
        grayscale_newaxis=True,
    )
    env = FrameStackObservation(env, 4)  # stack 4 frames
    env.reset(seed=seed)
    return env


#####

import torch
import torch.nn as nn

class AtariActorCritic(nn.Module):
    """Shared CNN backbone with separate actor and critic heads."""

    def __init__(self, n_actions: int):
        super().__init__()
        # Input: (batch, 4, 84, 84) - 4 stacked greyscale frames
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        cnn_out = 64 * 7 * 7  # 3136

        self.actor = nn.Sequential(
            nn.Linear(cnn_out, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(cnn_out, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        # x: (batch, 4, 84, 84), dtype float32
        feats = self.cnn(x)
        return self.actor(feats), self.critic(feats).squeeze(-1)
    

########

import torch.optim as optim
from torch.distributions import Categorical

def train_ppo(env_id, total_steps=5_000_000, horizon=1024,
              n_epochs=4, batch_size=128, lr=2.5e-4,
              gamma=0.99, gae_lambda=0.95,
              clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
              max_grad_norm=0.5, seed=42):

    env = make_env(env_id, seed=seed)
    n_actions = env.action_space.n
    model = AtariActorCritic(n_actions).to("cuda" if torch.cuda.
        is_available() else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    obs, _ = env.reset()
    episode_return, all_returns = 0.0, []

    for global_step in range(0, total_steps, horizon):
        # --- rollout collection ---
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], \
            [], [], [], []

        for _ in range(horizon):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(obs_t)
                dist = Categorical(logits=logits)
                action = dist.sample()

            obs_buf.append(obs_t.squeeze(0))
            act_buf.append(action)
            logp_buf.append(dist.log_prob(action))
            val_buf.append(value.squeeze())

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rew_buf.append(reward)
            done_buf.append(done)
            episode_return += reward

            if done:
                all_returns.append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset()

        # --- compute GAE advantages ---
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            _, next_val = model(obs_t)
            advantages, returns = compute_gae(
                rew_buf, val_buf, done_buf, next_val.item(), gamma,
                gae_lambda
            )

        # --- policy updates (K epochs) ---
        obs_t = torch.stack(obs_buf)
        act_t = torch.stack(act_buf)
        logp_t = torch.stack(logp_buf).detach()
        adv_t = (advantages - advantages.mean()) / (advantages.std() + \
            1e-8)
        ret_t = returns

        idx = torch.randperm(horizon)
        for _ in range(n_epochs):
            for start in range(0, horizon, batch_size):
                mb = idx[start:start + batch_size]
                logits, val_new = model(obs_t[mb])
                dist_new = Categorical(logits=logits)
                logp_new = dist_new.log_prob(act_t[mb])
                entropy = dist_new.entropy().mean()
                ratio = (logp_new - logp_t[mb]).exp()

                surr1 = ratio * adv_t[mb]
                surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t[mb]
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_vf = ((val_new - ret_t[mb]) ** 2).mean()
                loss = loss_pi + vf_coef * loss_vf - ent_coef * \
                    entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),
                    max_grad_norm)
                optimizer.step()

        if len(all_returns) % 10 == 0:
            print(f"step={global_step}  mean_ret={np.mean(all_returns[-100:]):.1f}")

    env.close()
    return model, all_returns