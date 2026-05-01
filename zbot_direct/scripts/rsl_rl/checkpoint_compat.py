# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility helpers for checkpoints saved by older RSL-RL releases."""

from __future__ import annotations

import torch


def load_runner_checkpoint_compat(runner, path: str):
    """Load either a current RSL-RL checkpoint or an older ActorCritic checkpoint.

    Older checkpoints store one ``model_state_dict`` containing ``actor.*``,
    ``critic.*``, and ``std`` keys. RSL-RL 4+ stores separate actor and critic
    model states, so playback needs a small key translation for legacy models.
    """
    checkpoint = torch.load(path, weights_only=False, map_location=runner.device)
    if "model_state_dict" not in checkpoint:
        return runner.load(path)

    model_state = checkpoint["model_state_dict"]
    actor_state = {}
    critic_state = {}
    for key, value in model_state.items():
        if key == "std":
            actor_state["distribution.std_param"] = value
        elif key.startswith("actor."):
            actor_state[f"mlp.{key.removeprefix('actor.')}"] = value
        elif key.startswith("critic."):
            critic_state[f"mlp.{key.removeprefix('critic.')}"] = value

    if not actor_state or not critic_state:
        raise ValueError(f"Could not translate legacy RSL-RL checkpoint: {path}")

    runner.alg.actor.load_state_dict(actor_state, strict=True)
    runner.alg.critic.load_state_dict(critic_state, strict=True)
    runner.current_learning_iteration = checkpoint.get("iter", 0)
    print("[INFO]: Loaded legacy RSL-RL ActorCritic checkpoint with converted actor/critic weights.")
    if "optimizer_state_dict" in checkpoint:
        print("[INFO]: Legacy optimizer state was not loaded because current RSL-RL uses separated actor/critic models.")
    return checkpoint.get("infos")
