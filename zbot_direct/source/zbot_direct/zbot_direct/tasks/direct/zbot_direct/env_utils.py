# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


@torch.jit.script
def compute_com(body_com: torch.Tensor, body_mass: torch.Tensor) -> torch.Tensor:
    mass_1d = body_mass.to(body_com.device)
    total_mass = float(torch.sum(mass_1d, dim=1)[0])
    mass = mass_1d.unsqueeze(-1).repeat(1, 1, 3)
    res = body_com * mass
    res = res / total_mass
    return torch.sum(res, dim=1).squeeze()
