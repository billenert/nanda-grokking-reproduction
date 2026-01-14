"""Activation caching using PyTorch hooks.

- run_with_cache(model, tokens, cache_spec) -> (logits, cache)
- Cache naming convention for residual stream, attention outputs, MLP outputs
"""

from typing import Any

import torch
import torch.nn as nn


def run_with_cache(
    model: nn.Module,
    tokens: torch.Tensor,
    cache_spec: list[str],
    cache_on_cpu: bool = True,
    cache_fp32: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run model forward pass and cache specified activations.

    Cache naming convention:
        - "resid_pre.l{L}": residual stream entering layer L (0-indexed)
        - "resid_post.l{L}": residual stream after layer L
        - "attn_out.l{L}": attention output (after output projection)
        - "mlp_out.l{L}": MLP output
        - "attn_weights.l{L}": attention weights [batch, heads, seq, seq]

    Args:
        model: Transformer model.
        tokens: Input tokens [batch, seq_len].
        cache_spec: List of cache names to capture.
        cache_on_cpu: If True, move cached tensors to CPU.
        cache_fp32: If True, cast cached tensors to float32.

    Returns:
        Tuple of (logits, cache) where cache is a dict of cached tensors.
    """
    cache: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, input, output):
            tensor = output
            if isinstance(output, tuple):
                tensor = output[0]

            tensor = tensor.detach()

            if cache_fp32 and tensor.dtype != torch.float32:
                tensor = tensor.float()

            if cache_on_cpu:
                tensor = tensor.cpu()

            cache[name] = tensor

        return hook

    # Register hooks based on cache_spec
    for name in cache_spec:
        if name.startswith("resid_pre.l"):
            layer_idx = int(name.split(".l")[1])
            if layer_idx == 0:
                # Hook on embedding output
                hooks.append(
                    model.token_embedding.register_forward_hook(make_hook(name))
                )
            else:
                # Hook on previous layer output
                hooks.append(
                    model.blocks[layer_idx - 1].register_forward_hook(make_hook(name))
                )

        elif name.startswith("resid_post.l"):
            layer_idx = int(name.split(".l")[1])
            hooks.append(
                model.blocks[layer_idx].register_forward_hook(make_hook(name))
            )

        elif name.startswith("attn_out.l"):
            layer_idx = int(name.split(".l")[1])
            hooks.append(
                model.blocks[layer_idx].attn.register_forward_hook(make_hook(name))
            )

        elif name.startswith("mlp_out.l"):
            layer_idx = int(name.split(".l")[1])
            hooks.append(
                model.blocks[layer_idx].mlp.register_forward_hook(make_hook(name))
            )

        elif name.startswith("attn_weights.l"):
            layer_idx = int(name.split(".l")[1])

            def make_attn_hook(cache_name: str):
                def hook(module, input, output):
                    # output is (attn_out, attn_weights) when return_attention=True
                    # We need to modify the forward call to return attention
                    pass
                return hook

            # For attention weights, we need to modify the model forward call
            # This is handled by the model's return_activations parameter

    try:
        # Run forward pass
        with torch.no_grad():
            logits = model(tokens)
    finally:
        # Remove all hooks
        for hook in hooks:
            hook.remove()

    return logits, cache


def get_cache_from_model(
    model: nn.Module,
    tokens: torch.Tensor,
    cache_on_cpu: bool = True,
    cache_fp32: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Get activations using model's built-in return_activations.

    This is simpler than hook-based caching for basic use cases.

    Args:
        model: Transformer model with return_activations support.
        tokens: Input tokens [batch, seq_len].
        cache_on_cpu: If True, move cached tensors to CPU.
        cache_fp32: If True, cast cached tensors to float32.

    Returns:
        Tuple of (logits, cache) where cache is a dict of cached tensors.
    """
    with torch.no_grad():
        logits, activations = model(tokens, return_activations=True)

    cache = {}
    for name, tensor in activations.items():
        if cache_fp32 and tensor.dtype != torch.float32:
            tensor = tensor.float()
        if cache_on_cpu:
            tensor = tensor.cpu()
        cache[name] = tensor

    return logits, cache
