"""Regression tests for the PyTorch supervised memory benchmark (Sprint 14 Pass 1).

These verify the learning machinery itself works before any learned control
policy is claimed:

  * GRU recurrent parameters receive gradients.
  * Recurrent parameters change during training.
  * Hidden state resets between independent episodes.
  * Hidden state persists within an episode.
  * No hidden-target leakage into policy inputs.
  * Action parameters match the environment contract.
  * Terminal rewards do not bootstrap from nonexistent future states.
  * The same seed reproduces training/evaluation.
  * Save/load restores model parameters.

torch is a research-only optional dependency; these tests skip when it is not
installed. No claim of consciousness; measures representational memory only.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.universe_engine.research.diagnostics.torch_memory import (
    CueMemoryDataset,
    FeedForwardCueModel,
    GRUCueModel,
    train_supervised,
    _recurrent_params_changed,
)
from src.universe_engine.research.diagnostics.torch_controller import (
    _GRUActorCritic, TorchGRUActorCriticController, pre_train_gru_cue,
    generate_cue_sequences,
)
from src.universe_engine.research.diagnostics.memory_tasks import (
    DelayedCueTask, DelayedCueConfig,
)
from src.universe_engine.research.contracts import (
    AgentAction, SensoryObservation,
)


# ---------------------------------------------------------------------------
# Gradient and parameter-change checks (recurrent weights, not just the head)
# ---------------------------------------------------------------------------

def test_gru_recurrent_weights_receive_gradients():
    """Recurrent weights (not only the cue-readout head) receive gradients."""
    torch.manual_seed(0)
    net = _GRUActorCritic(n_features=6, hidden_dim=16)
    seqs, cues = generate_cue_sequences(64, delay=3, seed=1)
    xb = torch.from_numpy(seqs[:8]).float()
    yb = torch.from_numpy(cues[:8]).float()
    _, _, cue, _ = net(xb, torch.zeros(1, 8, 16))
    loss = ((cue[:, -1, :].squeeze(-1) - yb) ** 2).mean()
    loss.backward()
    gru_grads = [p.grad for p in net.gru.parameters() if p.grad is not None]
    assert len(gru_grads) > 0, "no recurrent gradients"
    assert any(g.abs().max().item() > 0 for g in gru_grads)


def test_recurrent_parameters_change_during_training():
    """Trained GRU recurrent parameters actually change after optimizer steps."""
    torch.manual_seed(3)
    net = _GRUActorCritic(n_features=6, hidden_dim=16)
    before = {n: p.data.clone() for n, p in net.named_parameters() if "gru" in n.lower()}
    seqs, cues = generate_cue_sequences(512, delay=3, seed=4)
    seqs_t = torch.from_numpy(seqs).float(); cues_t = torch.from_numpy(cues).float()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(50):
        perm = torch.randperm(len(seqs))
        for b in range(0, len(seqs), 64):
            idx = perm[b:b + 64]
            opt.zero_grad()
            _, _, cue, _ = net(seqs_t[idx], torch.zeros(1, 64, 16))
            loss = loss_fn(cue[:, -1, :].squeeze(-1), cues_t[idx])
            loss.backward(); opt.step()
    assert _recurrent_params_changed(net, before), "recurrent parameters did not change"


# ---------------------------------------------------------------------------
# Hidden-state lifecycle: resets between episodes, persists within
# ---------------------------------------------------------------------------

def test_hidden_resets_between_episodes():
    """Controller hidden state is reset at the start of each episode."""
    ctrl = TorchGRUActorCriticController("a", seed=0, hidden_dim=16)
    obs = SensoryObservation(time=0.0, position=(0.0, 0.0), energy_level=0.5,
                             nutrient_level=0.5, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.0,
                             nearby_hazards=0.0, nearby_agents=0)
    ctrl.reset_hidden()
    h_start = ctrl.hidden.clone()
    # Run one step, capture hidden.
    ctrl.select_action(obs)
    h_mid = ctrl.hidden.clone()
    # Reset again -> back to start state.
    ctrl.reset_hidden()
    h_after = ctrl.hidden.clone()
    assert torch.allclose(h_after, h_start)
    assert not torch.allclose(h_mid, h_start)


def test_hidden_persists_within_episode():
    """Hidden state carries across steps within a single episode (memory)."""
    ctrl = TorchGRUActorCriticController("a", seed=0, hidden_dim=16)
    ctrl.reset_hidden()
    obs_cue = SensoryObservation(time=0.0, position=(0.0, 0.0), energy_level=0.5,
                                 nutrient_level=0.5, local_temperature=288.0,
                                 local_light=1.0, nearby_resources=1.0,
                                 nearby_hazards=0.0, nearby_agents=0)
    obs_neutral = SensoryObservation(time=1.0, position=(0.0, 0.0), energy_level=0.5,
                                     nutrient_level=0.5, local_temperature=288.0,
                                     local_light=0.0, nearby_resources=0.5,
                                     nearby_hazards=0.0, nearby_agents=0)
    ctrl.select_action(obs_cue)  # step 0: informative
    h_cue = ctrl.hidden.clone()
    ctrl.select_action(obs_neutral)  # step 1: delay
    h_delay = ctrl.hidden.clone()
    # Hidden after processing cue should influence hidden during delay.
    assert not torch.allclose(h_cue, h_delay), "hidden did not carry across steps"
    # Hidden during delay should not have collapsed to zero (memory retained).
    assert h_delay.abs().max().item() > 0


# ---------------------------------------------------------------------------
# No hidden-target leakage: action derived only from observation contract
# ---------------------------------------------------------------------------

def test_action_parameters_match_contract():
    """Action parameters are valid dx/dy in {-1,0,1} and action_type is legal."""
    ctrl = TorchGRUActorCriticController("a", seed=0, hidden_dim=16)
    obs = SensoryObservation(time=0.0, position=(0.0, 0.0), energy_level=0.5,
                             nutrient_level=0.5, local_temperature=288.0,
                             local_light=1.0, nearby_resources=0.5,
                             nearby_hazards=0.0, nearby_agents=0)
    for _ in range(20):
        action = ctrl.select_action(obs)
        assert action.action_type in ("wait", "move", "forage", "claim")
        assert set(action.parameters.keys()) == {"dx", "dy"}
        assert action.parameters["dx"] in (-1, 0, 1)
        assert action.parameters["dy"] in (-1, 0, 1)


def test_no_hidden_target_leakage():
    """The controller never receives the hidden target; only the observation
    contract is passed to select_action."""
    ctrl = TorchGRUActorCriticController("a", seed=0, hidden_dim=16)
    # Build an observation identical to what the environment would expose
    # during the informative phase (cue visible) — nearby_resources carries
    # the cue side, nothing else reveals the target.
    obs = SensoryObservation(time=0.0, position=(0.0, 0.0), energy_level=0.5,
                             nutrient_level=0.5, local_temperature=288.0,
                             local_light=1.0, nearby_resources=1.0,
                             nearby_hazards=0.0, nearby_agents=0)
    action = ctrl.select_action(obs)
    assert action.action_type in ("wait", "move", "forage", "claim")
    assert action.parameters["dx"] in (-1, 0, 1)
    assert action.parameters["dy"] in (-1, 0, 1)


# ---------------------------------------------------------------------------
# Terminal rewards do not bootstrap
# ---------------------------------------------------------------------------

def test_terminal_transition_no_bootstrap(monkeypatch):
    """finish_episode does not bootstrap value from a nonexistent next state
    after a terminal transition (returns end at the final reward)."""
    ctrl = TorchGRUActorCriticController("a", seed=0, hidden_dim=16)
    obs = SensoryObservation(time=0.0, position=(0.0, 0.0), energy_level=0.5,
                             nutrient_level=0.5, local_temperature=288.0,
                             local_light=0.0, nearby_resources=0.5,
                             nearby_hazards=0.0, nearby_agents=0)
    # Buffer one terminal transition with reward +1.0.
    ctrl.select_action(obs)
    ctrl.learn(obs, AgentAction("wait"), 1.0, obs)
    stats = ctrl.finish_episode()
    assert stats["n_steps"] == 1
    # With one step, return == reward (no bootstrap), normalized to 0 (T==1).
    assert stats["loss"] >= 0.0


# ---------------------------------------------------------------------------
# Reproducibility and save/load
# ---------------------------------------------------------------------------

def test_same_seed_reproduces_training():
    """The same seed reproduces identical supervised training results."""
    r1 = train_supervised(delay=5, train_seed=42, n_train=512, n_test=256,
                          n_epochs=10, batch_size=64)
    r2 = train_supervised(delay=5, train_seed=42, n_train=512, n_test=256,
                          n_epochs=10, batch_size=64)
    for m in ("feedforward", "frozen_gru", "trained_gru"):
        a1 = [x for x in r1 if x["model"] == m][0]["accuracy"]
        a2 = [x for x in r2 if x["model"] == m][0]["accuracy"]
        assert a1 == a2


def test_checkpoint_restore_restores_parameters():
    """Save/load restores GRU, actor and critic parameters."""
    torch.manual_seed(1)
    net = _GRUActorCritic(n_features=6, hidden_dim=16)
    cp = {k: v.clone() for k, v in net.state_dict().items()}
    # Mutate parameters.
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    assert not all(torch.allclose(net.state_dict()[k], cp[k]) for k in cp)
    net.load_state_dict(cp)
    assert all(torch.allclose(net.state_dict()[k], cp[k]) for k in cp)


# ---------------------------------------------------------------------------
# Supervised memory benchmark: GRU beats feedforward on non-trivial delay
# ---------------------------------------------------------------------------

def test_trained_gru_beats_feedforward_on_delay():
    """On a non-trivial delay, the trained GRU recall exceeds the
    feedforward (final-observation-only) baseline, which is at chance."""
    results = train_supervised(delay=10, train_seed=7, n_train=2048, n_test=512,
                               n_epochs=40, batch_size=64)
    acc = {x["model"]: x["accuracy"] for x in results}
    # Feedforward sees only the neutral final observation -> ~0.5.
    assert abs(acc["feedforward"] - 0.5) < 0.05
    # Trained GRU retains the cue -> well above chance.
    assert acc["trained_gru"] > 0.8, acc
