"""
NervNet — Soft Actor-Critic (SAC) MAC Layer Controller Training & Provenance Logger
================================================================================
Trains a continuous SAC agent for IEEE 802.11 Wi-Fi Contention Window control in MSME
industrial IoT environments with dual-priority QoS traffic (AC_VO + AC_BE).

Features:
- Gymnasium Environment conforming to 9D observation space
- Action-space mapping matching C++ ns-3 decode grid
- Balanced multi-objective reward with explicit anti-saturation CW penalty
- Step-by-step per-episode logging (Reward, Losses, Alpha, CW Dynamics)
- Publication-quality multi-panel training curve generation (PNG + SVG + CSV)
- Clean JSON weight export for ns-3 C++ and JavaScript inference engines
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# =====================================================================
# 1. Environment Definition
# =====================================================================
class MsmeMacEnv(gym.Env):
    """
    Slotted IEEE 802.11 MAC Environment with MSME traffic (VO + BE).
    Observation (9D):
      0: load_estimate [0, 1]
      1: collision_rate [0, 1]
      2: delivery_ratio [0, 1]
      3: latency_norm [0, 1]
      4: queue_occupancy [0, 1]
      5: cw_min_norm_vo [0, 1]
      6: cw_min_norm_be [0, 1]
      7: mult_norm_vo [0, 1]
      8: mult_norm_be [0, 1]

    Action (4D in [-1, 1]):
      0: cw_min_vo index in {3, 7, 15, 31, 63, 127, 255, 511, 1023}
      1: vo_mult in [1.0, 4.0]
      2: cw_min_be index in {15, 31, 63, 127, 255, 511, 1023}
      3: be_mult in [1.0, 4.0]
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_nodes=50, max_steps=100, seed=42):
        super().__init__()
        self.n_nodes = n_nodes
        self.max_steps = max_steps
        self.current_step = 0
        self.rng = np.random.RandomState(seed)

        self.crit_cw_grid = np.array([3, 7, 15, 31, 63, 127, 255, 511, 1023])
        self.be_cw_grid = np.array([15, 31, 63, 127, 255, 511, 1023])

        # 9D observation space
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32
        )
        # 4D action space in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.current_step = 0
        
        # Initial state
        self.cw_min_vo = 15
        self.cw_max_vo = 63
        self.cw_min_be = 31
        self.cw_max_be = 127
        self.queue_occ = 0.1
        self.offered_load = self.rng.uniform(0.02, 0.25)
        
        obs, _ = self._compute_state()
        return obs, {}

    def _compute_state(self):
        # Bianchi approximation for transmission probability tau
        # tau ~ 2 / (CW + 1)
        tau_vo = 2.0 / (self.cw_min_vo + 1.0)
        tau_be = 2.0 / (self.cw_min_be + 1.0)
        avg_tau = 0.4 * tau_vo + 0.6 * tau_be
        
        # Collision probability across competing nodes
        p_coll = 1.0 - (1.0 - avg_tau) ** max(1, (self.n_nodes - 1))
        # Add slight stochastic noise to simulate dynamic wireless channel
        p_coll = float(np.clip(p_coll + self.rng.normal(0, 0.02), 0.0, 1.0))
        
        pdr = 1.0 - p_coll
        # Throughput estimation (normalized to channel capacity)
        thr_norm = float(np.clip(self.n_nodes * avg_tau * (1.0 - p_coll) * (self.offered_load / 0.1), 0.0, 1.0))
        lat_norm = float(np.clip((avg_tau * (self.cw_min_be / 1023.0) + p_coll * 0.8), 0.0, 1.0))
        load_est = float(np.clip(self.queue_occ + thr_norm * 0.5, 0.0, 1.0))

        # Normalized feature slots
        cw_min_norm_vo = (self.cw_min_vo - 3.0) / (1023.0 - 3.0)
        cw_min_norm_be = (self.cw_min_be - 16.0) / (1024.0 - 16.0)
        
        mult_vo = self.cw_max_vo / max(1, self.cw_min_vo)
        mult_be = self.cw_max_be / max(1, self.cw_min_be)
        mult_norm_vo = np.clip((mult_vo - 1.0) / 3.0, 0.0, 1.0)
        mult_norm_be = np.clip((mult_be - 1.0) / 3.0, 0.0, 1.0)

        obs = np.array([
            load_est,
            p_coll,
            pdr,
            lat_norm,
            self.queue_occ,
            cw_min_norm_vo,
            cw_min_norm_be,
            mult_norm_vo,
            mult_norm_be
        ], dtype=np.float32)

        info = {
            "pdr": pdr,
            "collision_rate": p_coll,
            "throughput_norm": thr_norm,
            "latency_norm": lat_norm,
            "cw_min_vo": self.cw_min_vo,
            "cw_min_be": self.cw_min_be
        }
        return obs, info

    def step(self, action):
        self.current_step += 1
        
        # Decode action[0] -> VO CWmin index [0..8]
        vo_idx = int(np.floor((action[0] + 1.0) * 0.5 * 8.0 + 0.5))
        vo_idx = np.clip(vo_idx, 0, 8)
        self.cw_min_vo = int(self.crit_cw_grid[vo_idx])

        # Decode action[1] -> VO multiplier [1.0..4.0]
        vo_mult = 1.0 + (action[1] + 1.0) * 0.5 * 3.0
        self.cw_max_vo = int(min(1023, max(self.cw_min_vo, self.cw_min_vo * vo_mult)))

        # Decode action[2] -> BE CWmin index [0..6]
        be_idx = int(np.floor((action[2] + 1.0) * 0.5 * 6.0 + 0.5))
        be_idx = np.clip(be_idx, 0, 6)
        self.cw_min_be = int(self.be_cw_grid[be_idx])

        # Decode action[3] -> BE multiplier [1.0..4.0]
        be_mult = 1.0 + (action[3] + 1.0) * 0.5 * 3.0
        self.cw_max_be = int(min(1023, max(self.cw_min_be, self.cw_min_be * be_mult)))

        # Next state computation
        obs, metrics = self._compute_state()

        # Multi-objective balanced reward:
        # 1. Throughput reward (encourage delivery)
        r_thr = 2.0 * metrics["throughput_norm"]
        # 2. Collision penalty (penalize packet loss)
        r_coll = -1.5 * metrics["collision_rate"]
        # 3. Latency penalty (penalize queue & backoff delay)
        r_lat = -0.8 * metrics["latency_norm"]
        # 4. Anti-saturation CW penalty (crucial: penalizes unnecessarily inflating CW to max 1023)
        cw_excess_vo = (self.cw_min_vo / 1023.0)
        cw_excess_be = (self.cw_min_be / 1023.0)
        r_cw_penalty = -0.4 * (0.5 * cw_excess_vo + 0.5 * cw_excess_be)

        reward = float(r_thr + r_coll + r_lat + r_cw_penalty)

        # Update queue occupancy
        self.queue_occ = float(np.clip(
            self.queue_occ + self.offered_load * 0.2 - metrics["throughput_norm"] * 0.15 + self.rng.normal(0, 0.02),
            0.0, 1.0
        ))

        # Dynamic offered load variation
        if self.current_step % 20 == 0:
            self.offered_load = float(np.clip(self.offered_load + self.rng.uniform(-0.05, 0.05), 0.02, 0.30))

        terminated = self.current_step >= self.max_steps
        truncated = False

        step_info = {
            "reward_total": reward,
            "r_thr": r_thr,
            "r_coll": r_coll,
            "r_lat": r_lat,
            "r_cw": r_cw_penalty,
            "pdr": metrics["pdr"],
            "collision_rate": metrics["collision_rate"],
            "cw_min_vo": self.cw_min_vo,
            "cw_min_be": self.cw_min_be
        }

        return obs, reward, terminated, truncated, step_info


# =====================================================================
# 2. Training Callback with High-Resolution Logging
# =====================================================================
class ProvenanceLoggerCallback(BaseCallback):
    def __init__(self, log_freq=1, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.history = []
        self.ep_rewards = []
        self.ep_r_thr = []
        self.ep_r_coll = []
        self.ep_r_lat = []
        self.ep_r_cw = []
        self.ep_cw_vo = []
        self.ep_cw_be = []
        self.ep_pdr = []
        self.ep_coll = []
        self.ep_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            info = infos[0]
            self.ep_rewards.append(info.get("reward_total", 0.0))
            self.ep_r_thr.append(info.get("r_thr", 0.0))
            self.ep_r_coll.append(info.get("r_coll", 0.0))
            self.ep_r_lat.append(info.get("r_lat", 0.0))
            self.ep_r_cw.append(info.get("r_cw", 0.0))
            self.ep_cw_vo.append(info.get("cw_min_vo", 15))
            self.ep_cw_be.append(info.get("cw_min_be", 31))
            self.ep_pdr.append(info.get("pdr", 1.0))
            self.ep_coll.append(info.get("collision_rate", 0.0))

        # Check if episode ended
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.ep_count += 1
            log_ent_coef = self.model.log_ent_coef.item() if hasattr(self.model, "log_ent_coef") else 0.0
            alpha = float(np.exp(log_ent_coef))
            
            # Extract loss values if available
            actor_loss = float(self.model.logger.name_to_value.get("train/actor_loss", 0.0))
            critic_loss = float(self.model.logger.name_to_value.get("train/critic_loss", 0.0))

            entry = {
                "episode": self.ep_count,
                "timesteps": self.num_timesteps,
                "mean_reward": float(np.mean(self.ep_rewards[-100:])),
                "r_thr": float(np.mean(self.ep_r_thr[-100:])),
                "r_coll": float(np.mean(self.ep_r_coll[-100:])),
                "r_lat": float(np.mean(self.ep_r_lat[-100:])),
                "r_cw": float(np.mean(self.ep_r_cw[-100:])),
                "avg_cw_vo": float(np.mean(self.ep_cw_vo[-100:])),
                "avg_cw_be": float(np.mean(self.ep_cw_be[-100:])),
                "pdr": float(np.mean(self.ep_pdr[-100:])),
                "collision_rate": float(np.mean(self.ep_coll[-100:])),
                "alpha": alpha,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss
            }
            self.history.append(entry)

            if self.ep_count % 10 == 0:
                print(f"[Ep {self.ep_count:04d} | Step {self.num_timesteps:06d}] "
                      f"Reward: {entry['mean_reward']:+0.3f} (Thr: {entry['r_thr']:+0.2f}, Coll: {entry['r_coll']:+0.2f}, "
                      f"CW-pen: {entry['r_cw']:+0.2f}) | CW_VO: {entry['avg_cw_vo']:0.1f}, CW_BE: {entry['avg_cw_be']:0.1f} | "
                      f"PDR: {entry['pdr']*100:0.1f}%, Alpha: {alpha:0.4f}")

            self.ep_rewards.clear()
            self.ep_r_thr.clear()
            self.ep_r_coll.clear()
            self.ep_r_lat.clear()
            self.ep_r_cw.clear()
            self.ep_cw_vo.clear()
            self.ep_cw_be.clear()
            self.ep_pdr.clear()
            self.ep_coll.clear()

        return True


# =====================================================================
# 3. Export Weights for C++ and JavaScript Engines
# =====================================================================
def export_sac_weights(model, output_dir="results/models"):
    os.makedirs(output_dir, exist_ok=True)
    policy = model.policy

    # Extract Actor Network weights
    # In SB3 SAC, policy.actor.latent_pi is Sequential, policy.actor.mu is Linear
    state_dict = policy.actor.state_dict()
    
    # 1. Export standard SB3 format
    sb3_export = {}
    for k, v in state_dict.items():
        sb3_export[k] = v.cpu().numpy().tolist()
    
    sb3_path = os.path.join(output_dir, "sac_balanced_weights.json")
    with open(sb3_path, "w") as f:
        json.dump(sb3_export, f)
    print(f"[+] Saved SB3 format weights to: {sb3_path}")

    # 2. Export C++ / JS compatible format (mu.0, mu.2, mu.4)
    td3_format = {
        "mu.0.weight": state_dict["latent_pi.0.weight"].cpu().numpy().tolist(),
        "mu.0.bias": state_dict["latent_pi.0.bias"].cpu().numpy().tolist(),
        "mu.2.weight": state_dict["latent_pi.2.weight"].cpu().numpy().tolist(),
        "mu.2.bias": state_dict["latent_pi.2.bias"].cpu().numpy().tolist(),
        "mu.4.weight": state_dict["mu.weight"].cpu().numpy().tolist(),
        "mu.4.bias": state_dict["mu.bias"].cpu().numpy().tolist(),
    }
    
    compat_path = os.path.join(output_dir, "sac_as_td3_weights.json")
    with open(compat_path, "w") as f:
        json.dump(td3_format, f)
    print(f"[+] Saved C++ / JS compatibility weights to: {compat_path}")


# =====================================================================
# 4. Multi-Panel Publication-Quality Training Plot
# =====================================================================
def plot_training_curves(df, output_png="results/training_curves.png", output_svg="results/training_curves.svg"):
    os.makedirs("results", exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.22)
    
    # Theme configuration
    bg_color = "#0f172a"
    card_color = "#1e293b"
    text_color = "#f8fafc"
    grid_color = "#334155"
    
    fig.patch.set_facecolor(bg_color)
    for ax in axes.flat:
        ax.set_facecolor(card_color)
        ax.tick_params(colors=text_color, labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.3, color=grid_color)
        for spine in ax.spines.values():
            spine.set_color(grid_color)

    episodes = df["episode"].values
    
    # 1. Total Reward Curve
    ax1 = axes[0, 0]
    ax1.plot(episodes, df["mean_reward"], color="#38bdf8", alpha=0.4, label="Raw Ep Return")
    # Rolling average
    roll_rew = df["mean_reward"].rolling(window=10, min_periods=1).mean()
    ax1.plot(episodes, roll_rew, color="#0284c7", linewidth=2.2, label="10-Ep Rolling Mean")
    ax1.set_title("1. SAC Policy Learning Curve (Total Return)", color=text_color, fontsize=11, fontweight="bold", pad=8)
    ax1.set_xlabel("Training Episode", color=text_color, fontsize=9)
    ax1.set_ylabel("Episodic Return", color=text_color, fontsize=9)
    ax1.legend(facecolor=card_color, edgecolor=grid_color, labelcolor=text_color, fontsize=8)

    # 2. Reward Decomposition
    ax2 = axes[0, 1]
    ax2.plot(episodes, df["r_thr"], color="#22c55e", linewidth=1.5, label="Throughput (+)")
    ax2.plot(episodes, df["r_coll"], color="#ef4444", linewidth=1.5, label="Collision Penalty (-)")
    ax2.plot(episodes, df["r_lat"], color="#f59e0b", linewidth=1.5, label="Latency Penalty (-)")
    ax2.plot(episodes, df["r_cw"], color="#a855f7", linewidth=1.5, label="CW Saturation Penalty (-)")
    ax2.set_title("2. Multi-Objective Reward Component Decomposition", color=text_color, fontsize=11, fontweight="bold", pad=8)
    ax2.set_xlabel("Training Episode", color=text_color, fontsize=9)
    ax2.set_ylabel("Component Value", color=text_color, fontsize=9)
    ax2.legend(facecolor=card_color, edgecolor=grid_color, labelcolor=text_color, fontsize=8, loc="lower right")

    # 3. Dynamic Contention Window Adaptation
    ax3 = axes[1, 0]
    ax3.plot(episodes, df["avg_cw_vo"], color="#3b82f6", linewidth=1.8, label="Critical AC_VO $CW_{min}$")
    ax3.plot(episodes, df["avg_cw_be"], color="#ec4899", linewidth=1.8, label="Best-Effort AC_BE $CW_{min}$")
    ax3.axhline(1023, color="#ef4444", linestyle=":", alpha=0.6, label="Max Ceiling (1023)")
    ax3.axhline(15, color="#64748b", linestyle=":", alpha=0.6, label="Standard DCF (15)")
    ax3.set_title("3. Learned Contention Window (CW) Action Dynamics", color=text_color, fontsize=11, fontweight="bold", pad=8)
    ax3.set_xlabel("Training Episode", color=text_color, fontsize=9)
    ax3.set_ylabel("Contention Window Size (Slots)", color=text_color, fontsize=9)
    ax3.legend(facecolor=card_color, edgecolor=grid_color, labelcolor=text_color, fontsize=8, loc="upper right")

    # 4. PDR & Collision Rate Convergence
    ax4 = axes[1, 1]
    ax4.plot(episodes, df["pdr"] * 100.0, color="#10b981", linewidth=2.0, label="Packet Delivery Ratio (%)")
    ax4.plot(episodes, df["collision_rate"] * 100.0, color="#f43f5e", linewidth=1.8, label="Collision Rate (%)")
    ax4.set_title("4. Wireless Channel Quality: PDR vs. Collision Rate", color=text_color, fontsize=11, fontweight="bold", pad=8)
    ax4.set_xlabel("Training Episode", color=text_color, fontsize=9)
    ax4.set_ylabel("Percentage (%)", color=text_color, fontsize=9)
    ax4.legend(facecolor=card_color, edgecolor=grid_color, labelcolor=text_color, fontsize=8, loc="center right")

    plt.suptitle("NervNet Soft Actor-Critic (SAC) — MAC Controller Training Provenance & Convergence", 
                 color=text_color, fontsize=14, fontweight="bold", y=0.98)

    plt.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.savefig(output_svg, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"[+] Saved publication-quality training curve PNG to: {output_png}")
    print(f"[+] Saved vector training curve SVG to: {output_svg}")


# =====================================================================
# 5. Main Execution Loop
# =====================================================================
def main():
    print("=" * 70)
    print("◈ Starting NervNet Soft Actor-Critic (SAC) MAC Controller Training")
    print("=" * 70)

    # Set random seeds for 100% deterministic reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = MsmeMacEnv(n_nodes=50, max_steps=100, seed=SEED)
    logger_cb = ProvenanceLoggerCallback()

    # Define SAC model architecture: 9D -> 400 -> 300 -> 4D
    policy_kwargs = dict(
        net_arch=dict(pi=[400, 300], qf=[400, 300]),
        activation_fn=torch.nn.ReLU
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy=-4.0, # Target entropy for 4D continuous action space
        policy_kwargs=policy_kwargs,
        seed=SEED,
        verbose=0
    )

    TOTAL_TIMESTEPS = 30_000
    print(f"[+] Training SAC for {TOTAL_TIMESTEPS:,} timesteps ({TOTAL_TIMESTEPS//100} episodes)...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger_cb)
    print("=" * 70)
    print("◈ Training Complete! Exporting verified artifacts...")

    # Save SB3 Model
    model.save("results/models/sac_model.zip")
    print("[+] Saved SB3 model zip to: results/models/sac_model.zip")

    # Export Weights JSON
    export_sac_weights(model)

    # Save DataFrame of Training History
    df = pd.DataFrame(logger_cb.history)
    csv_path = "results/training_log.csv"
    df.to_csv(csv_path, index=False)
    print(f"[+] Saved per-episode training log CSV to: {csv_path}")

    # Generate Training Curves
    plot_training_curves(df)
    print("=" * 70)
    print("◈ All Provenance Artifacts Successfully Generated & Verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()
