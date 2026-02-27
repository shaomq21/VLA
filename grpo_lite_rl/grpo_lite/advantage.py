"""
Ultra-light GRPO advantage: batch-mean baseline, no per-prompt group, no index.
For n_samples=1 (one rollout per prompt). Minimal memory.
"""
import torch


def compute_grpo_lite_advantage(token_level_rewards: torch.Tensor,
                                eos_mask: torch.Tensor,
                                epsilon: float = 1e-6):
    """
    Args:
        token_level_rewards: (bs, response_length)
        eos_mask: (bs, response_length)
    Returns:
        advantages, returns: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    with torch.no_grad():
        mean = scores.mean()
        std = scores.std()
        adv = (scores - mean) / (std + epsilon)
        adv = adv.unsqueeze(-1).expand(-1, response_length) * eos_mask

    return adv, adv
