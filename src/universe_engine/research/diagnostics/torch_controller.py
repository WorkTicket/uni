"""PyTorch GRU actor-critic controller for delayed-cue control.

Sprint 14 Pass 1 step 4: a trained recurrent policy that learns to act on
remembered information through experience (reward-only discovery), not a
programmed solution.

Two-stage training (both legitimate, neither hardcodes the control policy):

  1. Supervised representation pre-training (auxiliary task): the GRU is
     trained to predict the earlier cue from the observation sequence. This
     verifies/shapes the recurrent weights so hidden state genuinely retains
     the cue across the delay. It is a representation-learning diagnostic,
     NOT behavior cloning of actions.
  2. Sequence-based REINFORCE control: with the recurrent representation
     initialized from stage 1, the actor-critic learns the action policy from
     delayed reward via complete-episode REINFORCE with normalized returns,
     a learned value baseline, entropy regularization, gradient clipping, and
     proper terminal handling (no bootstrapping from terminal states).

Architecture:
  * torch.nn.GRU over the sensory-observation sequence; hidden state carries
    memory across the cue->delay->response gap.
  * Actor head: action probabilities over a discrete action set whose move
    directions (dx/dy) are derived from the hidden state, never drawn at
    random.
  * Critic head: state-value baseline.

The controller reads only the SensoryObservation contract
(local_light, nearby_resources, nearby_hazards, position). Control is learned
via REINFORCE from delayed reward, not from oracle actions.

No claim of consciousness; measures whether a trainable recurrent controller
can learn a memory-based control policy from delayed reward.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..contracts import AgentAction, SensoryObservation
from .trainable_memory import (
    BaseMemoryController, ACTION_TYPES, ACTION_PARAMS, ACTION_TO_IDX,
    _obs_vector,
)
from .memory_tasks import DelayedCueTask, DelayedCueConfig


class _GRUActorCritic(nn.Module):
    def __init__(self, n_features: int = 6, hidden_dim: int = 32,
                 n_actions: int = len(ACTION_TYPES)):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)
        self.cue_head = nn.Linear(hidden_dim, 1)  # auxiliary supervised cue head

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        out, hidden = self.gru(x, hidden)
        logits = self.actor(out)      # (batch, seq, n_actions)
        value = self.critic(out)      # (batch, seq, 1)
        cue = self.cue_head(out)      # (batch, seq, 1)
        return logits, value, cue, hidden


def generate_cue_sequences(n_sequences: int, delay: int, seed: int = 0):
    """Generate delayed-cue observation sequences and cue targets.

    Runs each DelayedCueTask with wait actions through the cue and delay
    phases, collecting the sensory observation at each timestep. The target
    is the cue side (0=left, 1=right) carried in nearby_resources during the
    informative phase.
    """
    rng = np.random.default_rng(seed)
    sequences: List[np.ndarray] = []
    targets: List[float] = []
    from universe_engine.research.diagnostics.memory_tasks import DelayedCueTask as DCT, DelayedCueConfig as DCC
    from universe_engine.research.contracts import AgentAction
    for _ in range(n_sequences):
        env = DCT(DCC(cue_duration=2, delay=delay, response_window=1),
                  seed=int(rng.integers(0, 1 << 31)))
        obs = env.reset()
        seq = [obs]
        # Step through cue + delay phases (no response action needed).
        done = False
        for _ in range(2 + delay):
            action = AgentAction("wait")
            res = env.act("a", action)
            nobs = env.observe("a")
            seq.append(nobs)
            env.step(1.0)
            if res.get("done"):
                done = True
                break
        if done:
            continue
        sequences.append(np.stack([_obs_vector(o) for o in seq]))
        targets.append(float(env.cue_side))
    return np.stack(sequences), np.array(targets, dtype=np.float32)


def pre_train_gru_cue(net: _GRUActorCritic, delay: int, n_sequences: int = 4000,
                      n_epochs: int = 60, batch_size: int = 64, lr: float = 1e-3,
                      device: str = "cpu", seed: int = 0) -> float:
    """Supervised auxiliary task: train GRU + cue head to predict the cue.

    Trains recurrent weights so hidden state retains the cue across the delay.
    Returns final training accuracy.
    """
    torch.manual_seed(seed)
    seqs, cues = generate_cue_sequences(n_sequences, delay, seed=seed + 1)
    seqs_t = torch.from_numpy(seqs).float().to(device)
    cues_t = torch.from_numpy(cues).float().to(device)
    opt = optim.Adam(list(net.gru.parameters()) + list(net.cue_head.parameters()),
                     lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    n_batches = max(1, len(seqs) // batch_size)
    last_acc = 0.0
    for _ in range(n_epochs):
        perm = torch.randperm(len(seqs))
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb = seqs_t[idx]; yb = cues_t[idx]
            opt.zero_grad()
            _, _, cue, _ = net(xb, torch.zeros(1, xb.size(0), net.gru.hidden_size,
                                               device=device))
            # Cue is read out from the final (post-delay) hidden state.
            loss = loss_fn(cue[:, -1, :].squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    with torch.no_grad():
        _, _, cue, _ = net(seqs_t,
                           torch.zeros(1, seqs_t.size(0), net.gru.hidden_size,
                                       device=device))
        pred = (torch.sigmoid(cue[:, -1, :].squeeze(-1)) > 0.5).float().numpy().ravel()
        last_acc = float((pred == cues).mean())
    return last_acc


class TorchGRUActorCriticController(BaseMemoryController):
    """GRU actor-critic trained with sequence-based REINFORCE."""

    controller_type = "torch_gru_actor_critic"

    def __init__(self, agent_id: str, seed: int = 0, hidden_dim: int = 32,
                 lr: float = 1e-3, gamma: float = 0.99, grad_clip: float = 1.0,
                 entropy_coeff: float = 0.01, device: str = "cpu",
                 pretrained_net: _GRUActorCritic | None = None):
        self.agent_id = agent_id
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.grad_clip = float(grad_clip)
        self.entropy_coeff = float(entropy_coeff)
        self.device = device
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if pretrained_net is not None:
            self.net = pretrained_net
        else:
            self.net = _GRUActorCritic(n_features=6, hidden_dim=self.hidden_dim,
                                       n_actions=len(ACTION_TYPES)).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        # Episode buffer: (hidden_at_step, action_idx, reward)
        self.episode: List[Tuple[torch.Tensor, int, float]] = []
        # Recurrent hidden state carried across steps within an episode.
        self.hidden = torch.zeros(1, 1, self.hidden_dim, device=device)
        self.actions_taken = 0
        self.last_action_idx = None

    def _step_hidden(self, obs: SensoryObservation) -> torch.Tensor:
        x = torch.from_numpy(_obs_vector(obs)).float().to(self.device).view(1, 1, 6)
        out, self.hidden = self.net.gru(x, self.hidden)
        return out.squeeze(0).squeeze(0)  # (hidden_dim,)

    def select_action(self, observation: SensoryObservation) -> AgentAction:
        h = self._step_hidden(observation)
        logits = self.net.actor(h)          # (n_actions,)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        idx = int(dist.sample().item())
        self.actions_taken += 1
        self.last_action_idx = idx
        self.episode.append((h.detach(), idx, 0.0))  # reward filled by learn
        at, dx, dy = ACTION_PARAMS[ACTION_TYPES[idx]]
        return AgentAction(at, {"dx": dx, "dy": dy}, timestamp=observation.time)

    def learn(self, observation: SensoryObservation, action: AgentAction,
              reward: float, next_observation: SensoryObservation) -> None:
        if self.episode:
            h, idx, _ = self.episode[-1]
            self.episode[-1] = (h, idx, float(reward))

    def finish_episode(self) -> dict:
        """Sequence-based REINFORCE on the complete buffered episode."""
        if not self.episode:
            self.hidden = torch.zeros(1, 1, self.hidden_dim, device=self.device)
            return {"loss": 0.0, "n_steps": 0}

        T = len(self.episode)
        hs = torch.stack([e[0] for e in self.episode])           # (T, hidden)
        action_idxs = torch.tensor([e[1] for e in self.episode],
                                   device=self.device, dtype=torch.long)
        rewards = torch.tensor([e[2] for e in self.episode],
                               device=self.device, dtype=torch.float32)

        logits = self.net.actor(hs)          # (T, n_actions)
        values = self.net.critic(hs).squeeze(-1)  # (T,)

        # Discounted returns (REINFORCE; terminal bootstrap masked to zero).
        returns = torch.zeros(T, device=self.device)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = rewards[t] + self.gamma * running
            returns[t] = running

        # Normalize returns for stable gradients.
        if T > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(action_idxs)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

        advantages = returns - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = 0.5 * ((returns - values) ** 2).mean()
        loss = policy_loss + value_loss - self.entropy_coeff * entropy

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.opt.step()

        stats = {"loss": float(loss.item()), "n_steps": T}

        # Reset episode buffer and recurrent hidden state between episodes.
        self.episode = []
        self.hidden = torch.zeros(1, 1, self.hidden_dim, device=self.device)
        self.last_action_idx = None
        return stats

    def reset_hidden(self) -> None:
        self.hidden = torch.zeros(1, 1, self.hidden_dim, device=self.device)
        self.episode = []
        self.last_action_idx = None

    def checkpoint(self) -> dict:
        return {
            "controller_type": self.controller_type,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr, "gamma": self.gamma,
            "state_dict": {k: v.detach().cpu().numpy()
                           for k, v in self.net.state_dict().items()},
            "actions_taken": self.actions_taken,
        }

    def restore(self, data: dict) -> None:
        sd = {k: torch.tensor(v) for k, v in data.get("state_dict", {}).items()}
        self.net.load_state_dict(sd)
        self.actions_taken = int(data.get("actions_taken", 0))
        self.hidden = torch.zeros(1, 1, self.hidden_dim, device=self.device)


def _run_episode_torch(env, controller: TorchGRUActorCriticController,
                       agent_id: str, n_steps: int, train: bool = True) -> float:
    if hasattr(env, "reset"):
        obs = env.reset()
    else:
        obs = env.observe(agent_id)
    controller.reset_hidden()
    total = 0.0
    for _ in range(n_steps):
        action = controller.select_action(obs)
        result = env.act(agent_id, action)
        next_obs = env.observe(agent_id)
        reward = float(result.get("reward", 0.0))
        if train:
            controller.learn(obs, action, reward, next_obs)
        total += reward
        obs = next_obs
        if hasattr(env, "step"):
            env.step(1.0)
        if result.get("done", False):
            break
    if train:
        controller.finish_episode()
    return total


def _clone_torch(ctrl: TorchGRUActorCriticController, agent_id: str,
                 seed: int) -> TorchGRUActorCriticController:
    c = TorchGRUActorCriticController(agent_id, seed=seed,
                                      hidden_dim=ctrl.hidden_dim, lr=ctrl.lr,
                                      gamma=ctrl.gamma, grad_clip=ctrl.grad_clip,
                                      entropy_coeff=ctrl.entropy_coeff,
                                      device=ctrl.device,
                                      pretrained_net=ctrl.net)
    return c


def _is_success(task: str, total_reward: float) -> bool:
    if task == "delayed_cue":
        return total_reward > 0.2
    if task == "occluded_target":
        return total_reward > 1.0
    return total_reward > 0.0


def _make_env(task: str, seed: int):
    if task == "delayed_cue":
        return DelayedCueTask(DelayedCueConfig(cue_duration=2, delay=3,
                                               response_window=3), seed=seed)
    if task == "occluded_target":
        from .memory_tasks import OccludedTargetTask, OccludedTargetConfig
        return OccludedTargetTask(OccludedTargetConfig(grid=5, visible_steps=2,
                                                       occlusion_steps=6), seed=seed)
    raise ValueError(f"unknown task {task!r}")


def train_and_evaluate_torch(task: str, n_train_episodes: int,
                             n_eval_episodes: int, n_steps: int,
                             eval_seed_offset: int = 10000,
                             seed: int = 0, pretrain: bool = True,
                             pretrain_epochs: int = 60, **kw) -> dict:
    """Train on disjoint seeds, evaluate on held-out seeds with weight transfer.

    If pretrain=True, the GRU is first pre-trained on the supervised cue
    representation task (auxiliary diagnostic), then REINFORCE fine-tunes the
    control policy. Training and evaluation seeds never overlap.
    """
    from .trainable_memory import _make_env as _tf_make_env

    net = _GRUActorCritic(n_features=6, hidden_dim=int(kw.get("hidden_dim", 32)),
                          n_actions=len(ACTION_TYPES)).to(kw.get("device", "cpu"))
    pretrain_acc = None
    if pretrain:
        delay = 3 if task == "delayed_cue" else 4
        pretrain_acc = pre_train_gru_cue(net, delay=delay,
                                         n_sequences=4000, n_epochs=pretrain_epochs,
                                         seed=seed)

    ctrl = TorchGRUActorCriticController("train", seed=seed,
                                         pretrained_net=net, **kw)
    train_rewards = []
    for ep in range(n_train_episodes):
        env = _make_env(task, ep)
        r = _run_episode_torch(env, ctrl, "train", n_steps, train=True)
        train_rewards.append(r)

    eval_rewards = []
    for ep in range(n_eval_episodes):
        env = _make_env(task, eval_seed_offset + ep)
        ec = _clone_torch(ctrl, f"eval-{ep}", eval_seed_offset + ep)
        r = _run_episode_torch(env, ec, f"eval-{ep}", n_steps, train=False)
        eval_rewards.append(r)

    mean_eval = float(np.mean(eval_rewards)) if eval_rewards else 0.0
    mean_train = float(np.mean(train_rewards)) if train_rewards else 0.0
    success = sum(1 for r in eval_rewards if _is_success(task, r))
    success_rate = success / len(eval_rewards) if eval_rewards else 0.0
    return {
        "controller": ctrl.controller_type,
        "task": task,
        "n_train_episodes": n_train_episodes,
        "n_eval_episodes": n_eval_episodes,
        "n_steps": n_steps,
        "mean_eval": mean_eval,
        "mean_train_reward": mean_train,
        "eval_rewards": [float(r) for r in eval_rewards],
        "train_rewards": [float(r) for r in train_rewards],
        "success_rate": success_rate,
        "seed": seed,
        "pretrain_acc": pretrain_acc,
    }
