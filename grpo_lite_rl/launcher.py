#!/usr/bin/env python
"""
Launcher: patch grpo_lite into verl, then run verl.trainer.main_ppo.
Usage: python launcher.py [hydra overrides...]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIMPLEVLA_RL = ROOT.parent / "SimpleVLA-RL"

# Ensure grpo_lite_rl and SimpleVLA-RL are on path
for p in [str(ROOT), str(SIMPLEVLA_RL)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Patch before any verl trainer import
import grpo_lite.patch_verl
grpo_lite.patch_verl.patch_verl()

# Run main_ppo (will use patched compute_advantage and init_workers)
import runpy
runpy.run_module("verl.trainer.main_ppo", run_name="__main__")
