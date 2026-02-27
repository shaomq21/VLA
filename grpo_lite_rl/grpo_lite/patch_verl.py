"""
Patch SimpleVLA-RL verl to add grpo_lite without modifying its source.
Call patch_verl() before importing verl.trainer.main_ppo or similar.
"""
import sys
from pathlib import Path

# Add SimpleVLA-RL to path
SIMPLEVLA_RL = Path(__file__).resolve().parent.parent.parent / "SimpleVLA-RL"
if str(SIMPLEVLA_RL) not in sys.path:
    sys.path.insert(0, str(SIMPLEVLA_RL))


def patch_verl():
    """Inject grpo_lite into verl's core_algos and ray_trainer."""
    from grpo_lite.advantage import compute_grpo_lite_advantage
    import verl.trainer.ppo.core_algos as core_algos
    import verl.trainer.ppo.ray_trainer as ray_trainer
    import torch

    core_algos.compute_grpo_lite_advantage = compute_grpo_lite_advantage

    _original_compute_advantage = ray_trainer.compute_advantage

    def _patched_compute_advantage(data, gamma, lam, adv_estimator, config):
        if adv_estimator == 'grpo_lite':
            token_level_rewards = data.batch['token_level_rewards']
            responses = data.batch['responses']
            response_length = responses.size(1) * responses.size(2)
            finish_step = data.batch['finish_step'] * config.actor_rollout_ref.model.action_token_len
            steps = torch.arange(response_length, device=data.batch['responses'].device)
            steps_expanded = steps.unsqueeze(0).expand(data.batch['responses'].size(0), -1)
            response_mask = steps_expanded < finish_step.unsqueeze(1)
            advantages, returns = compute_grpo_lite_advantage(
                token_level_rewards=token_level_rewards, eos_mask=response_mask)
            data.batch['advantages'] = advantages
            data.batch['returns'] = returns
            return data
        return _original_compute_advantage(data, gamma, lam, adv_estimator, config)

    ray_trainer.compute_advantage = _patched_compute_advantage

    _orig_init = ray_trainer.RayTrainer.init_workers

    def _patched_init_workers(self):
        est = self.config.algorithm.adv_estimator
        if est == 'grpo_lite':
            self.config.algorithm.adv_estimator = 'grpo'
        try:
            return _orig_init(self)
        finally:
            self.config.algorithm.adv_estimator = est

    ray_trainer.RayTrainer.init_workers = _patched_init_workers
