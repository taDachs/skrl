from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import random
import sys
import time

import numpy as np

from skrl import config, logger


def set_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    """
    Set the seed for the random number generators

    .. note::

        In distributed runs, the worker/process seed will be incremented (counting from the defined value) according to its rank

    .. warning::

        Due to NumPy's legacy seeding constraint the seed must be between 0 and 2**32 - 1.
        Otherwise a NumPy exception (``ValueError: Seed must be between 0 and 2**32 - 1``) will be raised

    Modified packages:

    - random
    - numpy
    - torch (if available)
    - jax (skrl's PRNG key: ``config.jax.key``)

    Example::

        # fixed seed
        >>> from skrl.utils import set_seed
        >>> set_seed(42)
        [skrl:INFO] Seed: 42
        42

        # random seed
        >>> from skrl.utils import set_seed
        >>> set_seed()
        [skrl:INFO] Seed: 1776118066
        1776118066

        # enable deterministic. The following environment variables should be established:
        # - CUDA 10.1: CUDA_LAUNCH_BLOCKING=1
        # - CUDA 10.2 or later: CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:8
        >>> from skrl.utils import set_seed
        >>> set_seed(42, deterministic=True)
        [skrl:INFO] Seed: 42
        [skrl:WARNING] PyTorch/cuDNN deterministic algorithms are enabled. This may affect performance
        42

    :param seed: The seed to set. Is None, a random seed will be generated (default: ``None``)
    :type seed: int, optional
    :param deterministic: Whether PyTorch is configured to use deterministic algorithms (default: ``False``).
                          The following environment variables should be established for CUDA 10.1 (``CUDA_LAUNCH_BLOCKING=1``)
                          and for CUDA 10.2 or later (``CUBLAS_WORKSPACE_CONFIG=:16:8`` or ``CUBLAS_WORKSPACE_CONFIG=:4096:8``).
                          See PyTorch `Reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`_ for details
    :type deterministic: bool, optional

    :return: Seed
    :rtype: int
    """
    # generate a random seed
    if seed is None:
        try:
            seed = int.from_bytes(os.urandom(4), byteorder=sys.byteorder)
        except NotImplementedError:
            seed = int(time.time() * 1000)
        seed %= 2**31  # NumPy's legacy seeding seed must be between 0 and 2**32 - 1
    seed = int(seed)

    # set different seeds in distributed runs
    if config.torch.is_distributed:
        seed += config.torch.rank
    if config.jax.is_distributed:
        seed += config.jax.rank

    logger.info(f"Seed: {seed}")

    # numpy
    random.seed(seed)
    np.random.seed(seed)

    # torch
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            # On CUDA 10.1, set environment variable CUDA_LAUNCH_BLOCKING=1
            # On CUDA 10.2 or later, set environment variable CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:8

            logger.warning("PyTorch/cuDNN deterministic algorithms are enabled. This may affect performance")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"PyTorch seeding error: {e}")

    # jax
    config.jax.key = seed

    return seed

def categorical_td_loss(
    pred_log_probs: torch.Tensor,  # (n, num_bins)
    target_log_probs: torch.Tensor,  # (n, num_bins)
    reward: torch.Tensor,  # (n, 1)
    done: torch.Tensor,  # (n,)
    actor_log_probs: torch.Tensor,  # (n,)
    entropy_coefficient: torch.Tensor,  # (1,)
    gamma: float,
    num_bins: int,
    min_v: float,
    max_v: float,
    device: torch.device,
) -> tuple[torch.Tensor, Any]:
    with torch.no_grad():
        actor_entropy = actor_log_probs * entropy_coefficient

        bin_values = torch.linspace(min_v, max_v, num_bins, device=device).reshape(1, -1)
        target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
        target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

        b = (target_bin_values - min_v) / (max_v - min_v) * (num_bins - 1)
        l = torch.floor(b)
        u = torch.ceil(b)

        l_mask = F.one_hot(l.reshape(-1).long(), num_classes=num_bins).reshape(
            -1, num_bins, num_bins
        )
        u_mask = F.one_hot(u.reshape(-1).long(), num_classes=num_bins).reshape(
            -1, num_bins, num_bins
        )

        target_probs = torch.exp(target_log_probs)
        m_l = (target_probs * (u + (l == u).double() - b)).reshape(-1, num_bins, 1)
        m_u = (target_probs * (b - l)).reshape(-1, num_bins, 1)
        target_probs = torch.sum(m_l * l_mask + m_u * u_mask, axis=1)

    loss = -torch.mean(torch.sum(target_probs * pred_log_probs, axis=1))

    return loss, {"target_probs": target_probs}


def l2normalize_model(model: nn.Module):
    # Iterate over all modules in the model
    for name, module in model.named_modules():
        # Check if the module is a linear layer
        if "hyper_w" in name:
            # Normalize the weights using L2 norm
            with torch.no_grad():
                weight = module.weight
                normalized_weight = F.normalize(weight, p=2, dim=1)
                module.weight.copy_(normalized_weight)


