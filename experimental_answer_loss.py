"""Experimental answer-only cross entropy for causal language model distillation.

Standalone: not enabled in the Space or the production training loop. Offset pairs
must come from tokenizing the *combined* prompt and solution with the same tokenizer.
"""
import torch
import torch.nn.functional as F


def answer_token_indices(offsets, prompt_chars):
    """Indices of solution tokens wholly after the prompt, with a causal predecessor.

    Exclude boundary-crossing and zero-width tokens: neither is unambiguously an
    answer token. Truncated encodings with no solution tokens return an empty list.
    """
    return [i for i, (start, end) in enumerate(offsets)
            if i > 0 and start >= prompt_chars and end > start]


def answer_cross_entropy(logits, token_ids, offsets, prompt_chars, chunk_size=32):
    """Mean next-token CE over answer tokens; return (loss or None, token count).

    logits: (sequence, vocab), token_ids: (sequence,). Slicing before fp32 CE
    limits additional memory for large vocabularies. No implicit model forward.
    """
    if logits.ndim != 2 or token_ids.ndim != 1 or logits.shape[0] != token_ids.shape[0]:
        raise ValueError('expected logits (sequence, vocab) and matching token_ids')
    if len(offsets) != len(token_ids) or chunk_size <= 0:
        raise ValueError('offset count must match tokens and chunk_size must be positive')
    indices = answer_token_indices(offsets, prompt_chars)
    if not indices:
        return None, 0
    total = None
    for start in range(0, len(indices), chunk_size):
        target = torch.as_tensor(indices[start:start + chunk_size], device=logits.device)
        part = F.cross_entropy(logits[target - 1].float(), token_ids.to(logits.device)[target], reduction='sum')
        total = part if total is None else total + part
    return total / len(indices), len(indices)


def mixed_loss(distillation_loss, answer_loss, answer_weight):
    """Convex mixture; an absent answer is ignored, preserving distillation loss."""
    if not 0 <= answer_weight <= 1:
        raise ValueError('answer_weight must be between 0 and 1')
    if answer_loss is None or answer_weight == 0:
        return distillation_loss
    return (1 - answer_weight) * distillation_loss + answer_weight * answer_loss
