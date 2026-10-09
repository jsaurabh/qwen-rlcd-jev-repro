"""One-step expected-reward RL with a fixed SFT reference and optional CE anchor.

Enumerating the finite candidate actions gives the exact policy gradient of
expected reward, without sampled-action noise or generated text rollouts.
"""
def decision_rl_loss(logits, targets, reference_log_probs, counts, beta=0.05, ce_weight=0.25):
    import torch
    import torch.nn.functional as F
    if beta < 0 or ce_weight < 0:
        raise ValueError('Regularization weights must be nonnegative')
    counts = torch.as_tensor(counts,device=logits.device)
    if (counts < 1).any() or (counts > logits.shape[-1]).any():raise ValueError('Invalid candidate count')
    mask = torch.arange(logits.shape[-1],device=logits.device)[None] < torch.as_tensor(counts,device=logits.device)[:,None]
    if not mask.any(-1).all():raise ValueError('Empty candidate set')
    logp = F.log_softmax(logits.float().masked_fill(~mask, -1e9),dim=-1)
    p = logp.exp()
    truth = targets.float().detach()
    ref = reference_log_probs.float().detach()
    if not torch.allclose(torch.logsumexp(ref.masked_fill(~mask,-1e9),-1),torch.zeros(len(logits),device=logits.device),atol=2e-5):raise ValueError("Reference must be normalized")
    ref = ref.masked_fill(~mask,0)
    if not torch.isfinite(ref[mask]).all():raise ValueError('Invalid reference log probabilities')
    if not torch.allclose(truth.sum(-1),torch.ones(len(logits),device=logits.device),atol=1e-5):
        raise ValueError('Targets must be distributions')
    if (truth[~mask] != 0).any() or (truth < 0).any():raise ValueError('Invalid target support')
    # Correct action reward 1, incorrect action 0. For soft labels this is
    # expected correctness under the supplied label distribution.
    expected_reward = (p * truth).sum(-1)
    kl = torch.where(mask,p*(logp-ref),torch.zeros_like(p)).sum(-1)
    ce = -(truth*logp).sum(-1)
    per_row = -expected_reward + beta*kl + ce_weight*ce
    return per_row.mean(), {'reward':expected_reward.mean().detach(), 'kl':kl.mean().detach(), 'ce':ce.mean().detach()}
