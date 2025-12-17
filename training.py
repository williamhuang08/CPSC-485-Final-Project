# Import the necessary packages for the whole script
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, Independent, TransformedDistribution
from torch.distributions.transforms import TanhTransform
import gymnasium as gym
import mujoco
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import minari
from torch.utils.data import Dataset, DataLoader
import wandb
import os
from model import SkillPosterior, SkillPolicy, SkillPrior, MeanNetwork, StandardDeviationNetwork, TAWM

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# Loads the AntMaze dataset in Minari format
ant_maze_dataset = minari.load_dataset('D4RL/antmaze/medium-diverse-v1', download=True)

print(ant_maze_dataset[0].actions.shape)
print(ant_maze_dataset[0].observations.keys())
print(ant_maze_dataset[0].observations["observation"].shape)
print(ant_maze_dataset[0].observations["achieved_goal"].shape)

# B, the number of subtrajectories per batch 
B = 100

# T, the length of each subtrajectory
T = 40

# AntMaze state and action dims (from Minari)
state_dim = 29
action_dim = 8

# Initialize the models
q_phi = SkillPosterior(state_dim=state_dim, action_dim=action_dim).to(device)
pi_theta = SkillPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
p_psi = TAWM(state_dim=state_dim).to(device)
p_omega = SkillPrior(state_dim=state_dim).to(device)


def make_episode_splits(minari_dataset, train=0.8, val=0.1, test=0.1, seed=0):
    """Return three lists of episode indices (train_ids, val_ids, test_ids)."""
    episodes = list(minari_dataset.iterate_episodes())
    n = len(episodes)
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    train_ids = idxs[:n_train]
    val_ids = idxs[n_train:n_train+n_val]
    test_ids = idxs[n_train+n_val:]
    return train_ids, val_ids, test_ids

class SubtrajDataset(Dataset):
    """
    Loops over minari_dataset.iterate_episodes(), but keeps only episodes whose index is in episode_ids
    """
    def __init__(self, minari_dataset, T, episode_ids, stride=3):
        self.T = T
        self.items = []  

        # Iterate all episodes but only process those whose global index is in episode_ids
        for ep_idx, ep in enumerate(minari_dataset.iterate_episodes()):
            if ep_idx not in set(episode_ids):
                continue
            obs = ep.observations["observation"]          
            ach = ep.observations["achieved_goal"]        
            act = ep.actions                               
            Ltot = len(obs)
            if Ltot < T + 1:
                continue

            state_ext = np.concatenate([obs, ach], axis=-1).astype(np.float32)
            for t in range(0, Ltot - T, stride):
                state_seq = state_ext[t:t+T]         
                s0 = state_seq[0]             
                action_seq = act[t:t+T].astype(np.float32)  
                sT = state_ext[t+T]           
                self.items.append((s0, state_seq, action_seq, sT))

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, i):
        """standardize s0, state_sequence, and sT by (x - mean) / std"""
        
        s0, S, A, sT = self.items[i]
        if hasattr(self, "stats") and self.stats is not None:
            S_mean, S_std = self.stats
            S  = (S  - S_mean) / S_std
            s0 = (s0 - S_mean) / S_std
            sT = (sT - S_mean) / S_std
            A  = A
        return {
            "s0": torch.as_tensor(s0, dtype=torch.float32),
            "state_sequence": torch.as_tensor(S, dtype=torch.float32),
            "action_sequence": torch.as_tensor(A, dtype=torch.float32),
            "sT": torch.as_tensor(sT, dtype=torch.float32),
        }

def collate(batch):
    return {
        "s0": torch.stack([b["s0"] for b in batch], 0),
        "state_sequence": torch.stack([b["state_sequence"] for b in batch], 0),
        "action_sequence": torch.stack([b["action_sequence"] for b in batch], 0),
        "sT": torch.stack([b["sT"] for b in batch], 0),
    }


# Pick indices for train/test/split
train_ids, val_ids, test_ids = make_episode_splits(ant_maze_dataset, train=0.8, val=0.0, test=0.2, seed=0)
print(f"train:{len(train_ids)}  val:{len(val_ids)}  test:{len(test_ids)}")

# Datasets from episode subsets
train_ds = SubtrajDataset(ant_maze_dataset, T=T, episode_ids=train_ids, stride=3)
val_ds = SubtrajDataset(ant_maze_dataset, T=T, episode_ids=val_ids,   stride=3)
test_ds = SubtrajDataset(ant_maze_dataset, T=T, episode_ids=test_ids,  stride=3)  

print(f"train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}")

# find per-feature mean and std from all state_sequence timesteps in train_ds
def compute_stats(ds):
    Ss = []
    for item in ds.items:
        Ss.append(item[1])  # state_sequence [T,29]
    S = np.concatenate([x.reshape(-1, x.shape[-1]) for x in Ss], axis=0)
    S_mean, S_std = S.mean(0), S.std(0) + 1e-6
    return (S_mean, S_std)

S_mean, S_std = 0, 1

# pass stats into datasets
train_ds.stats = (S_mean, S_std)
val_ds.stats = (S_mean, S_std)

train_loader = DataLoader(train_ds, batch_size=B, shuffle=True,  collate_fn=collate, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=B, shuffle=False, collate_fn=collate, drop_last=False)

test_ds.stats = (S_mean, S_std)
test_loader = DataLoader(test_ds, batch_size=B, shuffle=False, collate_fn=collate, drop_last=False)

beta = 1.0
alpha = 1.0  


def e_terms(batch):
    s0, S, A = batch["s0"], batch["state_sequence"], batch["action_sequence"]
    B, T, _  = S.shape
    denom = B * T

    # Posterior q_phi(z|tau)
    mu_q, std_q = q_phi(S, A)  # [B, Z_DIM]
    z = mu_q + std_q * torch.randn_like(mu_q)      

    # Low-level policy pi_theta(a|s,z)
    z_bt = z.unsqueeze(1).expand(B, T, -1)        
    mu_pi, std_pi = pi_theta(
        S.reshape(B*T, -1),
        z_bt.reshape(B*T, -1)
    )
    mu_pi, std_pi = mu_pi.view(B, T, -1), std_pi.view(B, T, -1)

    a_dist = Independent(Normal(mu_pi, std_pi), 1)        
    post_dist = Independent(Normal(mu_q,  std_q), 1)
    mu_pr, std_pr = p_omega(s0)                              
    prior_dist = Independent(Normal(mu_pr, std_pr), 1)

    log_pi = a_dist.log_prob(A).sum() / denom      
    log_prior = prior_dist.log_prob(z).sum() / denom
    log_post = post_dist.log_prob(z).sum() / denom

    E_loss = -log_pi - beta * log_prior + beta * log_post
    return {"e_loss": E_loss,"log_pi": log_pi,"log_prior": log_prior,"log_post": log_post
    }
def m_terms(batch):
    s0, S, A, sT = batch["s0"], batch["state_sequence"], batch["action_sequence"], batch["sT"]
    B, T, _  = S.shape
    denom = B * T

    mu_q, std_q = q_phi(S, A)
    z = (mu_q + std_q * torch.randn_like(mu_q)).detach()

    z_bt = z.unsqueeze(1).expand(B, T, -1)
    mu_pi, std_pi = pi_theta(S.reshape(B*T, -1), z_bt.reshape(B*T, -1))
    mu_pi, std_pi = mu_pi.view(B, T, -1), std_pi.view(B, T, -1)

    a_dist = Independent(Normal(mu_pi, std_pi), 1)

    mu_T, std_T = p_psi(s0, z)
    sT_dist = Independent(Normal(mu_T, std_T), 1)

    mu_pr, std_pr = p_omega(s0)
    prior_dist = Independent(Normal(mu_pr, std_pr), 1)

    sT_loss = -sT_dist.log_prob(sT).sum() / denom
    a_loss = -a_dist.log_prob(A).sum() / denom
    prior_loss = -prior_dist.log_prob(z).sum()/ denom

    M_loss = alpha * sT_loss + a_loss + beta * prior_loss
    return {"m_loss": M_loss, "sT_loss": sT_loss, "a_loss": a_loss, "prior_loss": prior_loss}



@torch.no_grad()
def eval_epoch(val_loader, q_phi, pi_theta, p_psi, p_omega, device):
    """Compute validation E- and M-loss"""
    q_phi.eval()
    pi_theta.eval()
    p_psi.eval()
    p_omega.eval()
    e_sum, m_sum, n = 0.0, 0.0, 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        terms = e_terms(batch)
        e = terms["e_loss"]
        # e = compute_e_loss(batch)
        terms = m_terms(batch)
        m = terms["m_loss"]
        # m = compute_m_loss(batch)
        e_sum += float(e.item())
        m_sum += float(m.item())
        n += 1
    if n == 0: 
        return None, None
    return e_sum / n, m_sum / n

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

def skill_model_training_with_val(
    train_loader, val_loader,
    q_phi, pi_theta, p_psi, p_omega,
    e_lr=5e-5, m_lr=5e-5,
    epochs=50, e_steps=1, m_steps=1, grad_clip=1.0
):
    q_phi.to(device)
    pi_theta.to(device)
    p_psi.to(device)
    p_omega.to(device)

    opt = torch.optim.Adam([{"params": q_phi.parameters(), "lr": e_lr},{"params": list(pi_theta.parameters()) + list(p_psi.parameters()) + list(p_omega.parameters()), "lr": m_lr},
    ])

    tr_e, tr_m, va_e, va_m = [], [], [], []

    for epoch in range(1, epochs+1):
        e_run = m_run = 0.0 # Running e_loss, m_loss, in current epoch

        nb = 0

        for batch in train_loader:
            # Rebuilds dictionary but moves tensors to the device
            batch = {k: v.to(device) for k, v in batch.items()}
            nb += 1

            # E step: update q_phi
            # In E-step, train the posterior while freezing other parameters
            q_phi.train()
            pi_theta.eval()
            p_psi.eval()
            p_omega.eval()
            set_requires_grad(q_phi, True)
            set_requires_grad(pi_theta, False)
            set_requires_grad(p_psi, False)
            set_requires_grad(p_omega, False)

            for _ in range(e_steps):
                opt.zero_grad(set_to_none=True)
                e_loss = e_terms(batch)["e_loss"]
                e_loss.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(q_phi.parameters(), grad_clip)
                opt.step()
            e_run += float(e_loss.item())

            # M step: update theta, psi, omega
            # Freeze posterior weights, update all other weights

            q_phi.eval()
            pi_theta.train()
            p_psi.train()
            p_omega.train()
            set_requires_grad(q_phi, False)
            set_requires_grad(pi_theta, True)
            set_requires_grad(p_psi, True)
            set_requires_grad(p_omega, True)

            for _ in range(m_steps):
                # Reset gradients
                opt.zero_grad(set_to_none=True)
                terms = m_terms(batch)
                m_loss = terms["m_loss"]
                m_loss.backward()

                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(list(pi_theta.parameters()) + list(p_psi.parameters()) + list(p_omega.parameters()),grad_clip)
                opt.step()
            m_run += float(m_loss.item())

        # Calculate the average losses over all the batches in the epoch
        e_epoch = e_run / max(1, nb)
        m_epoch = m_run / max(1, nb)
        tr_e.append(e_epoch)
        tr_m.append(m_epoch)

        # validation
        ve, vm = eval_epoch(val_loader, q_phi, pi_theta, p_psi, p_omega, device)
        va_e.append(ve); va_m.append(vm)

        print(f"[Epoch {epoch:03d}/{epochs}] "
              f"train E:{e_epoch:.4f}  M:{m_epoch:.4f} "
              f"| val E:{ve:.4f}  M:{vm:.4f}")

        wandb.log({
            "train/E_loss": e_epoch,
            "train/M_loss": m_epoch,
            "val/E_loss": ve,
            "val/M_loss": vm,
            "epoch": epoch
        }, step=epoch)

    plt.figure(figsize=(7.5,4.5))
    plt.plot(tr_e, label="Train E-loss")
    plt.plot(tr_m, label="Train M-loss")
    if all(v is not None for v in va_e):
        plt.plot(va_e, label="Val E-loss")
    if all(v is not None for v in va_m):
        plt.plot(va_m, label="Val M-loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("EM training: train vs. val losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    fig = plt.gcf()
    wandb.log({"plots/loss_curves": wandb.Image(fig)}, step=epoch)
    plt.close(fig)

    return {"train_E": tr_e, "train_M": tr_m, "val_E": va_e, "val_M": va_m}



wandb.init(
    project="tawm-skill-learning",
    name=f"antmaze-medium_beta{beta}",
    config=dict(
        B=B, T=T, Z_DIM=Z_DIM, NUM_NEURONS=NUM_NEURONS,
        e_lr=5e-5, m_lr=5e-5, e_steps=1, m_steps=1,
        dataset="D4RL/antmaze/medium-diverse-v1",
        device=device
    )
)

# wandb.watch([q_phi, pi_theta, p_psi, p_omega], log="gradients", log_freq=200)

curves = skill_model_training_with_val(train_loader, test_loader, q_phi, pi_theta, p_psi, p_omega, epochs=250, e_lr=5e-5, m_lr=5e-5, e_steps=1, m_steps=1)


wandb.finish()



# Load and save the model to a .pth file
def save_checkpoint(path, q_phi, pi_theta, p_psi, p_omega):
    ckpt = {
        "q_phi": q_phi.state_dict(),
        "pi_theta": pi_theta.state_dict(),
        "p_psi": p_psi.state_dict(),
        "p_omega": p_omega.state_dict(),
        "S_stats": {"mean": S_mean, "std": S_std},
        "config": dict(B=B, T=T, Z_DIM=Z_DIM, NUM_NEURONS=NUM_NEURONS,device=str(device))
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)
    print(f"checkpoint saved -> {path}")

def load_checkpoint(path, q_phi, pi_theta, p_psi, p_omega, strict=True):
    ckpt = torch.load(path)
    q_phi.load_state_dict(ckpt["q_phi"], strict=strict)
    pi_theta.load_state_dict(ckpt["pi_theta"], strict=strict)
    p_psi.load_state_dict(ckpt["p_psi"], strict=strict)
    p_omega.load_state_dict(ckpt["p_omega"], strict=strict)
    stats = ckpt.get("S_stats", None)
    if stats is not None:
        global S_mean, S_std
        S_mean, S_std = stats["mean"], stats["std"]
    print(f"[checkpoint] loaded <- {path}")
    return ckpt

save_checkpoint("checkpoints/antmaze_diverse_detached_250_1.pth", q_phi, pi_theta, p_psi, p_omega)




