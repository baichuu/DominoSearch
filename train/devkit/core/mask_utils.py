"""Strict checkpoint loading and persistent parameter masks for pruning."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must be a state_dict or contain 'state_dict'.")
    return {name.removeprefix("module."): value for name, value in state_dict.items()}


def load_initial_checkpoint(path, model):
    """Load an initialization checkpoint and reject partial model matches."""
    checkpoint = torch.load(Path(path), map_location="cpu")
    incompatible = model.load_state_dict(_state_dict_from_checkpoint(checkpoint), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Initial checkpoint does not exactly match the model. "
            "Missing: {}; unexpected: {}".format(
                list(incompatible.missing_keys)[:5],
                list(incompatible.unexpected_keys)[:5],
            )
        )


def load_parameter_masks(path, model, device) -> Dict[str, torch.Tensor]:
    """Load and validate a parameter-name -> binary mask dictionary."""
    payload = torch.load(Path(path), map_location="cpu")
    masks = payload.get("parameter_masks", payload) if isinstance(payload, dict) else payload
    if not isinstance(masks, dict):
        raise ValueError("Mask file must be a dictionary or contain 'parameter_masks'.")

    parameters = dict(model.named_parameters())
    unknown = sorted(set(masks) - set(parameters))
    if unknown:
        raise ValueError("Mask file contains unknown parameters: {}".format(unknown[:5]))
    if not masks:
        raise ValueError("Mask file contains no parameter masks.")

    validated = {}
    for name, mask in masks.items():
        if not torch.is_tensor(mask):
            raise ValueError("Mask for '{}' is not a tensor.".format(name))
        if tuple(mask.shape) != tuple(parameters[name].shape):
            raise ValueError(
                "Mask shape mismatch for '{}': {} != {}".format(
                    name, tuple(mask.shape), tuple(parameters[name].shape)
                )
            )
        binary = mask.to(device=device, dtype=parameters[name].dtype)
        if not torch.all((binary == 0) | (binary == 1)):
            raise ValueError("Mask for '{}' is not binary.".format(name))
        validated[name] = binary
    return validated


def apply_parameter_masks(model, masks) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, mask in masks.items():
            parameters[name].mul_(mask)


def register_parameter_mask_hooks(model, masks):
    """Mask gradients and return removable hook handles."""
    parameters = dict(model.named_parameters())
    handles = []
    for name, mask in masks.items():
        handles.append(parameters[name].register_hook(lambda gradient, mask=mask: gradient * mask))
    return handles
